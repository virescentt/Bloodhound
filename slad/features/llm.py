"""LLM (API zgodne z OpenAI: Ollama, vLLM, llama.cpp, hostowane modele open-weights) — wzbogacanie cech tekstu.

    LLMClient.from_settings(cfg: LLMSettings, cache_dir) -> LLMClient | None
        None, gdy LLM wyłączony, brak base_url/model albo serwer nieosiągalny (szybka próba GET /models, <= 5 s;
        wynik próby zapamiętany na PROBE_TTL_S, więc kolejne konteksty ekstrakcji nie czekają ponownie).
    LLMClient.complete_json(system, user, schema=None) -> dict      (LLMError przy niepowodzeniu)
    enrich(text, client) -> dict | None
        {persons, brands, product, promise, cta, narratives, keywords} — oczyszczone; nigdy nie rzuca wyjątku.

Odpowiedź JSON: response_format json_schema → json_object → schemat w prompcie (tryb obniżany automatycznie,
gdy serwer go nie obsługuje), extra_body z ustawień (np. {"reasoning_effort": "none"} dla Ollamy), timeout,
tolerancyjne parsowanie (bloki <think>, płotki ```json). Pamięć podręczna na dysku: klucz = sha256 (model, system,
user, schema) — powtórne przetworzenie zbioru nie odpytuje modelu. Po serii błędów połączenia klient na chwilę
przestaje odpytywać serwer (pipeline nie czeka na każdy timeout).
Przeniesione i dostosowane z projektu Odwołania2JSON (odwolania2json/llm/client.py).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MODES = ("json_schema", "json_object", "prompt")
PROBE_TIMEOUT_S = 5.0
PROBE_TTL_S = 300.0
FAILURES_BEFORE_PAUSE = 3
PAUSE_S = 300.0
DEFAULT_MAX_CHARS = 4000
MAX_LIST_ITEMS = 10

_PROBES: dict[tuple[str, str, str], tuple[float, str | None]] = {}
_PROBES_LOCK = threading.Lock()
# tryb JSON obsługiwany przez serwer (base_url, model) — ustalony raz na proces, kolejni klienci nie próbują od nowa
_LEARNED_MODES: dict[tuple[str, str], str] = {}


class LLMError(RuntimeError):
    pass


def parse_json(text: str) -> dict[str, Any]:
    """Obiekt JSON z odpowiedzi modelu (usuwa bloki <think>…</think> i płotki ```json)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL).strip()  # niedomknięty początek rozumowania
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("brak obiektu JSON w odpowiedzi")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("odpowiedź nie jest obiektem JSON")
    return data


def _is_unsupported_format(e: Exception) -> bool:
    """Serwer odrzucił tryb JSON → schodzimy niżej (json_schema → json_object → prompt).
    Np. runner MLX Ollamy odpowiada 501 „structured output is unavailable”."""
    if getattr(e, "status_code", None) == 501:
        return True
    msg = str(e).lower()
    return any(
        k in msg for k in ("response_format", "json_schema", "not supported", "structured output", "unsupported")
    )


def _is_connection_error(e: Exception) -> bool:
    try:
        from openai import APIConnectionError, APITimeoutError

        if isinstance(e, APIConnectionError | APITimeoutError):
            return True
    except ImportError:  # pragma: no cover
        pass
    return isinstance(e, ConnectionError | TimeoutError)


class LLMClient:
    def __init__(self, settings: Any, cache_dir: str | Path | None = None, client: Any = None):
        self.settings = settings
        self.model: str = settings.model or ""
        # pola spoza kontraktu LLMSettings — opcjonalne (getattr z domyślnymi wartościami)
        self._server_key = (str(getattr(settings, "base_url", "") or ""), self.model)
        mode = getattr(settings, "structured_output", None) or "json_schema"
        self.mode: str = _LEARNED_MODES.get(self._server_key) or (mode if mode in MODES else "json_schema")
        self.temperature = float(getattr(settings, "temperature", 0.0) or 0.0)
        self.max_tokens = int(getattr(settings, "max_tokens", 1024) or 1024)
        self.max_retries = int(getattr(settings, "max_retries", 1) or 0)
        self.max_chars = int(getattr(settings, "max_chars", DEFAULT_MAX_CHARS) or DEFAULT_MAX_CHARS)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=settings.base_url,
                api_key=settings.api_key or "not-needed",
                timeout=float(getattr(settings, "timeout_s", 60.0) or 60.0),
                max_retries=1,
            )
        self.client = client
        self._lock = threading.Lock()
        self._failures = 0
        self._paused_until = 0.0

    # ------------------------------------------------------------------ tworzenie i dostępność
    @classmethod
    def from_settings(cls, cfg: Any, cache_dir: str | Path | None = None, *, probe: bool = True) -> LLMClient | None:
        """Klient albo None (LLM wyłączony / nieskonfigurowany / nieosiągalny) — wtedy działają heurystyki."""
        if cfg is None or not getattr(cfg, "enabled", False):
            return None
        if not getattr(cfg, "base_url", None) or not getattr(cfg, "model", None):
            log.info("LLM nieskonfigurowany (brak LLM_BASE_URL / LLM_MODEL w .env) — używam heurystyk")
            return None
        try:
            client = cls(cfg, cache_dir)
        except Exception as e:  # noqa: BLE001
            log.warning("Klient LLM niedostępny: %s — używam heurystyk", e)
            return None
        if probe:
            problem = client.check_available()
            if problem:
                log.warning("LLM wyłączony (%s): %s — używam heurystyk", cfg.base_url, problem)
                return None
        return client

    def check_available(self) -> str | None:
        """None, gdy LLM jest używalny; inaczej powód po polsku. Wynik zapamiętywany na PROBE_TTL_S."""
        key = (str(self.settings.base_url), self.model, str(getattr(self.settings, "api_key", "") or ""))
        now = time.monotonic()
        with _PROBES_LOCK:
            hit = _PROBES.get(key)
            if hit is not None and now - hit[0] < PROBE_TTL_S:
                return hit[1]
        problem = self._probe()
        with _PROBES_LOCK:
            _PROBES[key] = (time.monotonic(), problem)
        return problem

    def _probe(self) -> str | None:
        """Jedno szybkie GET /models. Nieznany błąd (serwer bez /models) → zakładamy, że działa."""
        try:
            from openai import APIConnectionError, APITimeoutError, AuthenticationError, PermissionDeniedError
        except ImportError as e:  # pragma: no cover
            return f"brak pakietu openai ({e})"
        try:
            page = self.client.with_options(timeout=PROBE_TIMEOUT_S, max_retries=0).models.list()
        except (APIConnectionError, APITimeoutError):
            return "serwer nie odpowiada"
        except (AuthenticationError, PermissionDeniedError):
            return "błąd autoryzacji (sprawdź LLM_API_KEY w .env)"
        except Exception:  # noqa: BLE001
            return None
        try:
            ids = {str(m.id).lower() for m in page.data}
        except Exception:  # noqa: BLE001
            return None
        wanted = {self.model.lower(), f"{self.model}:latest".lower()}  # Ollama: „nazwa” == „nazwa:latest”
        if ids and not wanted & ids:
            shown = ", ".join(sorted(ids)[:10])
            return f"model „{self.model}” niedostępny na serwerze (dostępne: {shown})"
        return None

    # ------------------------------------------------------------------ pamięć podręczna
    def _key(self, *parts: Any) -> str:
        raw = json.dumps([self.model, *parts], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_read(self, key: str) -> dict[str, Any] | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def _cache_write(self, key: str, data: dict[str, Any]) -> None:
        if self.cache_dir is None:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.cache_dir, prefix=".tmp-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self.cache_dir / f"{key}.json")
        except OSError as e:
            log.debug("Nie zapisano odpowiedzi LLM w cache: %s", e)

    # ------------------------------------------------------------------ zapytanie
    def _note_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= FAILURES_BEFORE_PAUSE:
                self._paused_until = time.monotonic() + PAUSE_S
                self._failures = 0
                log.warning(
                    "LLM: %d błędy połączenia z rzędu — przerwa %.0f s (heurystyki)", FAILURES_BEFORE_PAUSE, PAUSE_S
                )

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any] | None = None, name: str = "output"
    ) -> dict[str, Any]:
        key = self._key(system, user, schema)
        cached = self._cache_read(key)
        if cached is not None:
            return cached
        if time.monotonic() < self._paused_until:
            raise LLMError("LLM chwilowo niedostępny (seria błędów połączenia)")

        messages: list[dict[str, str]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        parse_retries = 0
        while True:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            extra_body = getattr(self.settings, "extra_body", None)
            if extra_body:
                kwargs["extra_body"] = dict(extra_body)
            mode = self.mode
            if mode == "json_schema" and schema is not None:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": name, "schema": schema, "strict": False},
                }
            elif mode in ("json_schema", "json_object"):
                kwargs["response_format"] = {"type": "json_object"}
            elif schema is not None and not messages[0]["content"].endswith("</schema>"):
                messages[0] = {
                    "role": "system",
                    "content": messages[0]["content"]
                    + "\n\nOdpowiedz wyłącznie obiektem JSON zgodnym ze schematem:\n<schema>\n"
                    + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
                    + "\n</schema>",
                }
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001
                if _is_unsupported_format(e) and mode != "prompt":
                    self.mode = MODES[MODES.index(mode) + 1]
                    _LEARNED_MODES[self._server_key] = self.mode
                    log.info("Serwer LLM odrzucił response_format — przechodzę na tryb „%s”", self.mode)
                    continue
                if _is_connection_error(e):
                    self._note_failure()
                raise LLMError(f"zapytanie do LLM nie powiodło się: {e}") from e
            with self._lock:
                self._failures = 0
            try:
                content = resp.choices[0].message.content or ""
            except (AttributeError, IndexError, TypeError):
                content = ""
            try:
                data = parse_json(content)
            except ValueError as e:
                if parse_retries >= self.max_retries:
                    raise LLMError(f"LLM nie zwrócił poprawnego JSON: {e}") from e
                parse_retries += 1
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"Błąd parsowania JSON: {e}. Zwróć wyłącznie poprawny obiekt JSON."},
                ]
                continue
            self._cache_write(key, data)
            return data


# ------------------------------------------------------------------ wzbogacanie cech tekstu

SYSTEM_PROMPT = """Jesteś analitykiem zespołu CERT badającym fałszywe reklamy i strony oszustów internetowych (scam) w Polsce.
Otrzymasz tekst jednego materiału: treść reklamy, tekst odczytany z obrazu (OCR — może zawierać błędy) i/lub tekst
strony docelowej. Wypełnij pola JSON WYŁĄCZNIE na podstawie tego tekstu — nie zgaduj i nie dopisuj nic spoza materiału.

- persons: znane osoby publiczne, których nazwisko lub wizerunek wykorzystano (celebryci, sportowcy, politycy,
  przedsiębiorcy, dziennikarze). Pełne imię i nazwisko w mianowniku, np. „Robert Lewandowski”. Bez fikcyjnych
  „klientów”, „ekspertów” i autorów opinii.
- brands: podszywane lub przywoływane media, banki, firmy, instytucje i serwisy, np. „TVN24”, „PKO BP”, „Orlen”,
  „ZUS”, „InPost”. Nazwa w mianowniku.
- product: nazwa promowanego produktu, platformy, aplikacji lub „systemu” (np. platformy inwestycyjnej); null, gdy brak.
- promise: obiecywana korzyść, krótko, z kwotami, np. „5000 zł dziennie”; null, gdy brak.
- cta: wezwanie do działania, krótki cytat, np. „Zarejestruj się teraz”; null, gdy brak.
- narratives: 0–3 najlepiej pasujące tagi narracji, wyłącznie z listy poniżej (sam identyfikator tagu).
- keywords: do 10 słów kluczowych opisujących materiał, małymi literami, po polsku.

Dozwolone tagi narracji:
{tags}

Odpowiedz wyłącznie obiektem JSON."""


def _allowed_narratives() -> dict[str, str]:
    try:
        from slad.features.text import narrative_labels

        return narrative_labels()
    except Exception as e:  # noqa: BLE001
        log.warning("Nie wczytano listy narracji: %s", e)
        return {}


def enrichment_schema(tags: list[str]) -> dict[str, Any]:
    narrative_item: dict[str, Any] = {"type": "string"}
    if tags:
        narrative_item["enum"] = tags
    nullable = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {
            "persons": {"type": "array", "items": {"type": "string"}},
            "brands": {"type": "array", "items": {"type": "string"}},
            "product": nullable,
            "promise": nullable,
            "cta": nullable,
            "narratives": {"type": "array", "items": narrative_item},
            "keywords": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_LIST_ITEMS},
        },
        "required": ["persons", "brands", "product", "promise", "cta", "narratives", "keywords"],
        "additionalProperties": False,
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v for v in re.split(r"[,;\n]", value) if v.strip()]
    if isinstance(value, list | tuple):
        return list(value)
    return []


def _clean_str(value: Any, max_len: int) -> str | None:
    if isinstance(value, list | tuple):
        value = ", ".join(str(v) for v in value if v)
    if value is None or isinstance(value, dict | bool):
        return None
    text = " ".join(str(value).split()).strip(" \"'„”")
    if not text or text.lower() in {"null", "none", "brak", "n/a", "-"}:
        return None
    return text[:max_len]


def _clean_names(value: Any, max_len: int = 80) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in _as_list(value):
        name = _clean_str(item, 1000)
        if not name or len(name) > max_len:
            continue
        if name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
        if len(out) >= MAX_LIST_ITEMS:
            break
    return out


def _tag_key(text: str) -> str:
    from slad.features.text import fold

    return re.sub(r"[\s\-]+", "_", fold(text)).strip("_")


def clean_enrichment(raw: dict[str, Any], allowed: dict[str, str]) -> dict[str, Any]:
    """Odpowiedź modelu → bezpieczny słownik (typy, długości, tylko dozwolone tagi narracji)."""
    by_key = {_tag_key(tag): tag for tag in allowed} | {_tag_key(label): tag for tag, label in allowed.items()}
    narratives: list[str] = []
    for item in _as_list(raw.get("narratives")):
        tag = by_key.get(_tag_key(str(item)))
        if tag and tag not in narratives:
            narratives.append(tag)
    keywords: list[str] = []
    for item in _as_list(raw.get("keywords")):
        kw = _clean_str(item, 40)
        if kw and kw.lower() not in keywords:
            keywords.append(kw.lower())
    return {
        "persons": _clean_names(raw.get("persons")),
        "brands": _clean_names(raw.get("brands")),
        "product": _clean_str(raw.get("product"), 200),
        "promise": _clean_str(raw.get("promise"), 200),
        "cta": _clean_str(raw.get("cta"), 200),
        "narratives": narratives[:3],
        "keywords": keywords[:MAX_LIST_ITEMS],
    }


def enrich(text: str, client: Any) -> dict[str, Any] | None:
    """Cechy materiału odczytane przez LLM (oczyszczone) albo None — nigdy nie rzuca wyjątku."""
    if client is None or not isinstance(text, str) or not text.strip():
        return None
    try:
        allowed = _allowed_narratives()
        tags = "\n".join(f"- {tag}: {label}" for tag, label in allowed.items()) or "- (brak)"
        max_chars = int(
            getattr(client, "max_chars", None)
            or getattr(getattr(client, "settings", None), "max_chars", 0)
            or DEFAULT_MAX_CHARS
        )
        user = "Materiał do analizy:\n<<<\n" + text.strip()[:max_chars] + "\n>>>"
        raw = client.complete_json(
            SYSTEM_PROMPT.format(tags=tags), user, schema=enrichment_schema(list(allowed)), name="analiza_reklamy"
        )
        if not isinstance(raw, dict):
            return None
        return clean_enrichment(raw, allowed)
    except Exception as e:  # noqa: BLE001 — LLM jest opcjonalny, pipeline działa dalej na heurystykach
        log.warning("LLM: wzbogacanie cech nie powiodło się: %s", e)
        return None
