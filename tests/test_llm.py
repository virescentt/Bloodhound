"""Klient LLM i wzbogacanie cech — bez sieci (atrapy klienta OpenAI)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:  # bez instalacji pakietu / conftest.py
    sys.path.insert(0, str(ROOT))

from slad.features import ExtractContext, llm  # noqa: E402
from slad.features import iocs as iocs_module  # noqa: E402
from slad.features.llm import LLMClient, LLMError, clean_enrichment, enrich, parse_json  # noqa: E402
from slad.features.text import narrative_labels, text_features  # noqa: E402
from slad.models import Material, SourceKind  # noqa: E402
from slad.settings import LLMSettings, load_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(llm, "_PROBES", {})
    monkeypatch.setattr(llm, "_LEARNED_MODES", {})
    monkeypatch.setattr(iocs_module, "_INFRA", {"registered_domain": None, "normalize_url": None})


def settings(**kw) -> LLMSettings:
    base = {"enabled": True, "base_url": "http://llm.test/v1", "model": "test-model", "extra_body": {"x": 1}}
    return LLMSettings(**(base | kw))


def answer(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeOpenAI:
    """Atrapa openai.OpenAI: chat.completions.create zwraca kolejne odpowiedzi lub rzuca podane wyjątki."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(json.loads(json.dumps(kwargs, default=str)))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return answer(item)


class Unsupported(Exception):
    status_code = 501


# ------------------------------------------------------------------ parsowanie


@pytest.mark.parametrize(
    "content",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '<think>rozważam {"b": 2}</think>\n{"a": 1}',
        'rozumowanie bez otwarcia</think>{"a": 1}',
        'Oto wynik: {"a": 1} — gotowe.',
    ],
)
def test_parse_json_tolerant(content):
    assert parse_json(content) == {"a": 1}


@pytest.mark.parametrize("content", ["", "brak json", "[1, 2]", "{niepoprawny"])
def test_parse_json_errors(content):
    with pytest.raises(ValueError):
        parse_json(content)


# ------------------------------------------------------------------ tworzenie klienta


def test_from_settings_returns_none_when_disabled_or_unconfigured(tmp_path):
    assert LLMClient.from_settings(settings(enabled=False), tmp_path) is None
    assert LLMClient.from_settings(settings(base_url=None), tmp_path) is None
    assert LLMClient.from_settings(settings(model=None), tmp_path) is None
    assert LLMClient.from_settings(None, tmp_path) is None


def test_from_settings_probe_is_remembered(tmp_path, monkeypatch):
    calls = []

    def probe(self):
        calls.append(1)
        return "serwer nie odpowiada"

    monkeypatch.setattr(LLMClient, "_probe", probe)
    assert LLMClient.from_settings(settings(), tmp_path) is None
    assert LLMClient.from_settings(settings(), tmp_path) is None
    assert len(calls) == 1  # drugi kontekst nie czeka ponownie na nieosiągalny serwer

    monkeypatch.setattr(LLMClient, "_probe", lambda self: None)
    client = LLMClient.from_settings(settings(model="inny-model"), tmp_path)
    assert isinstance(client, LLMClient) and client.cache_dir == tmp_path
    assert LLMClient.from_settings(settings(), tmp_path, probe=False) is not None


def test_extract_context_falls_back_to_heuristics_when_llm_unreachable(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMClient, "_probe", lambda self: "serwer nie odpowiada")
    cfg = load_settings(llm={"enabled": True, "base_url": "http://llm.test/v1", "model": "m"})
    ctx = ExtractContext.from_settings(cfg, tmp_path, with_faces=False)
    assert ctx.llm is None


# ------------------------------------------------------------------ complete_json


SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}}


def test_complete_json_uses_schema_extra_body_and_disk_cache(tmp_path):
    fake = FakeOpenAI('{"a": 1}')
    client = LLMClient(settings(), tmp_path, client=fake)
    assert client.complete_json("system", "user", SCHEMA) == {"a": 1}
    sent = fake.calls[0]
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["schema"] == SCHEMA
    assert sent["extra_body"] == {"x": 1} and sent["model"] == "test-model"
    assert len(list(tmp_path.glob("*.json"))) == 1
    # ten sam materiał ponownie: z dysku, także w nowym kliencie (powtórne przetworzenie zbioru)
    again = LLMClient(settings(), tmp_path, client=FakeOpenAI())
    assert again.complete_json("system", "user", SCHEMA) == {"a": 1}


def test_complete_json_downgrades_unsupported_modes_and_remembers(tmp_path):
    fake = FakeOpenAI(Unsupported("structured output is unavailable"), Unsupported("unsupported"), '{"a": 2}')
    client = LLMClient(settings(), None, client=fake)
    assert client.complete_json("system", "user", SCHEMA) == {"a": 2}
    assert [c.get("response_format", {}).get("type") for c in fake.calls] == ["json_schema", "json_object", None]
    assert "<schema>" in fake.calls[2]["messages"][0]["content"]
    assert client.mode == "prompt"
    assert LLMClient(settings(), None, client=FakeOpenAI()).mode == "prompt"  # kolejny klient nie próbuje od nowa


def test_complete_json_retries_on_invalid_json(tmp_path):
    fake = FakeOpenAI("to nie jest JSON", '{"a": 3}')
    client = LLMClient(settings(), None, client=fake)
    assert client.complete_json("s", "u", SCHEMA) == {"a": 3}
    assert "Błąd parsowania JSON" in fake.calls[1]["messages"][-1]["content"]
    with pytest.raises(LLMError):
        LLMClient(settings(), None, client=FakeOpenAI("zle", "znowu zle")).complete_json("s", "u2", SCHEMA)


def test_connection_errors_pause_the_client():
    fake = FakeOpenAI(*[ConnectionError("odmowa połączenia")] * 3)
    client = LLMClient(settings(), None, client=fake)
    for i in range(3):
        with pytest.raises(LLMError):
            client.complete_json("s", f"u{i}")
    with pytest.raises(LLMError, match="chwilowo niedostępny"):
        client.complete_json("s", "u-next")
    assert len(fake.calls) == 3  # w czasie przerwy serwer nie jest odpytywany


# ------------------------------------------------------------------ enrich


class FakeLLM:
    """Atrapa LLMClient na poziomie complete_json (jak w text_features)."""

    def __init__(self, result=None, error: Exception | None = None, max_chars: int = 4000):
        self.result, self.error, self.max_chars = result, error, max_chars
        self.seen: list[tuple[str, str, dict]] = []

    def complete_json(self, system, user, schema=None, name="output"):
        self.seen.append((system, user, schema))
        if self.error:
            raise self.error
        return self.result


DIRTY = {
    "persons": ["Robert Lewandowski", "robert lewandowski", "", None, "x" * 200, "Jan Kowalski"],
    "brands": "PKO Bank Polski, Narodowy Bank Polski",
    "product": "null",
    "promise": "  5000 zł   dziennie ",
    "cta": ["Zarejestruj się", "teraz"],
    "narratives": ["inwestycja_krypto", "Fałszywy wywiad / artykuł ze znaną osobą", "nieznany_tag", "bot-ai"],
    "keywords": ["Bitcoin", "bitcoin", "ZYSK"] + [f"slowo{i}" for i in range(20)],
}


def test_clean_enrichment():
    out = clean_enrichment(DIRTY, narrative_labels())
    assert out["persons"] == ["Robert Lewandowski", "Jan Kowalski"]
    assert out["brands"] == ["PKO Bank Polski", "Narodowy Bank Polski"]
    assert out["product"] is None
    assert out["promise"] == "5000 zł dziennie"
    assert out["cta"] == "Zarejestruj się, teraz"
    assert out["narratives"] == ["inwestycja_krypto", "falszywy_wywiad", "bot_ai"]
    assert out["keywords"][:2] == ["bitcoin", "zysk"] and len(out["keywords"]) == 10


def test_enrich_builds_prompt_and_truncates():
    fake = FakeLLM(DIRTY, max_chars=50)
    out = enrich("A" * 500, fake)
    assert out is not None and out["narratives"][0] == "inwestycja_krypto"
    system, user, schema = fake.seen[0]
    assert "inwestycja_krypto" in system and "Dozwolone tagi narracji" in system
    assert user.count("A") == 50
    assert set(schema["properties"]["narratives"]["items"]["enum"]) == set(narrative_labels())


def test_enrich_never_raises():
    assert enrich("tekst", FakeLLM(error=LLMError("timeout"))) is None
    assert enrich("tekst", FakeLLM(error=RuntimeError("cokolwiek"))) is None
    assert enrich("tekst", FakeLLM(result=["nie", "słownik"])) is None
    assert enrich("tekst", None) is None
    assert enrich("   ", FakeLLM(DIRTY)) is None


# ------------------------------------------------------------------ łączenie z heurystykami w text_features


def test_text_features_merges_llm(tmp_path):
    ctx = ExtractContext(settings=load_settings(), cache_root=tmp_path)
    ctx.llm = FakeLLM(
        {
            "persons": ["robert lewandowski", "Lewandowskiego", "Jan Kowalski"],
            "brands": ["PKO Bank Polski", "TVN24"],
            "product": "Quantum AI",
            "promise": "5000 zł dziennie",
            "cta": "Zarejestruj się",
            "narratives": ["inwestycja_krypto", "nieznany_tag"],
            "keywords": ["bitcoin"],
        }
    )
    m = Material(id="m1", source=SourceKind.TEXT, text="Robert Lewandowski poleca. Klienci PKO BP już zarabiają.")
    tf = text_features(m, [], "", ctx)
    assert tf.llm_used is True
    assert tf.persons == ["Robert Lewandowski", "Jan Kowalski"]  # kanonizacja + bez duplikatów
    assert tf.brands == ["PKO BP", "TVN24"]
    assert "inwestycja_krypto" in tf.narratives and "nieznany_tag" not in tf.narratives
    assert tf.product == "Quantum AI" and tf.promise == "5000 zł dziennie" and tf.cta == "Zarejestruj się"
    system, user, _ = ctx.llm.seen[0]
    assert "[Tekst reklamy / opis materiału]" in user and "Robert Lewandowski poleca" in user


def test_text_features_keeps_heuristics_when_llm_fails(tmp_path):
    ctx = ExtractContext(settings=load_settings(), cache_root=tmp_path)
    ctx.llm = FakeLLM(error=LLMError("serwer nie odpowiada"))
    m = Material(id="m1", source=SourceKind.TEXT, text="Zarejestruj się! Robert Lewandowski zarabia 5000 zł dziennie.")
    tf = text_features(m, [], "", ctx)
    assert tf.llm_used is False
    assert tf.persons == ["Robert Lewandowski"] and tf.cta and tf.promise == "zarabia 5000 zł dziennie"
