"""Cechy tekstowe (bez LLM) i OCR: normalizacja, słowniki osób/marek, narracje, obietnica/CTA/produkt, MinHash."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:  # bez instalacji pakietu / conftest.py
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from slad.features import ExtractContext, ocr  # noqa: E402
from slad.features import iocs as iocs_module  # noqa: E402
from slad.features.text import (  # noqa: E402
    canonical_name,
    detect_lang,
    extract_keywords,
    find_cta,
    find_product,
    find_promise,
    fold,
    load_gazetteer,
    load_narratives,
    match_gazetteer,
    match_narratives,
    minhash_jaccard,
    minhash_object,
    minhash_signature,
    narrative_labels,
    normalize_text,
    text_features,
)
from slad.models import Material, SourceKind  # noqa: E402
from slad.settings import TextSettings, load_settings  # noqa: E402


class DictCache(dict):
    """Zastępnik JsonCache (get/set) — testy nie zależą od modułu infrastruktury."""

    def set(self, key, value):
        self[key] = value


@pytest.fixture(autouse=True)
def _no_infra(monkeypatch):
    monkeypatch.setattr(iocs_module, "_INFRA", {"registered_domain": None, "normalize_url": None})


@pytest.fixture
def ctx(tmp_path):
    c = ExtractContext(settings=load_settings(), cache_root=tmp_path)
    c._caches["ocr"] = DictCache()
    assert c.llm is None
    return c


def material(text: str | None, **kw) -> Material:
    return Material(id=kw.pop("id", "m1"), source=SourceKind.TEXT, text=text, **kw)


def persons_in(text: str, ocr_text: str = "") -> list[str]:
    return match_gazetteer(load_gazetteer("persons"), fold(text), text, fold(ocr_text))


def brands_in(text: str) -> list[str]:
    return match_gazetteer(load_gazetteer("brands"), fold(text), text)


# ------------------------------------------------------------------ normalizacja


def test_normalize_text():
    raw = "„Zarabiaj”\u200b  z\u00a0Orlenem — 𝐙𝐀𝐑𝐀𝐁𝐈𝐀𝐉 «teraz»\n\tŻÓŁĆ"
    assert normalize_text(raw) == '"zarabiaj" z orlenem - zarabiaj "teraz" żółć'
    assert normalize_text(None) == "" and normalize_text("") == ""


def test_homoglyphs_fixed_only_in_mixed_words():
    assert normalize_text("Bіtcoin") == "bitcoin"  # ukraińskie „і” w łacińskim słowie
    assert normalize_text("Слава Україні") == "слава україні"  # prawdziwa cyrylica zostaje


def test_fold_removes_polish_diacritics():
    assert fold("Rafał Brzoską, ŻÓŁĆ, gęślą jaźń") == "rafal brzoska, zolc, gesla jazn"


# ------------------------------------------------------------------ słowniki osób i marek


def test_gazetteer_files_load():
    persons, brands = load_gazetteer("persons"), load_gazetteer("brands")
    assert len(persons) >= 30 and len(brands) >= 40
    assert {"Robert Lewandowski", "Rafał Brzoska", "Elon Musk"} <= {e.name for e in persons}
    assert {"PKO BP", "Orlen", "InPost", "ZUS", "Baltic Pipe"} <= {e.name for e in brands}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Wywiad z Robertem Lewandowskim", ["Robert Lewandowski"]),
        ("Lewandowskiego zapytano o inwestycje", ["Robert Lewandowski"]),
        ("Anna Lewandowska zdradza sekret", ["Anna Lewandowska"]),
        ("Rafał Brzoska ujawnia, a Brzoski nikt nie zatrzyma", ["Rafał Brzoska"]),
        ("rafal brzoska i krzysztof ibisz", ["Rafał Brzoska", "Krzysztof Ibisz"]),
        ("Premier Tusk i Donalda Tuska", ["Donald Tusk"]),
        ("Igi Świątek nie było", ["Iga Świątek"]),
        ("Dawid Podsiadły zaśpiewał", ["Dawid Podsiadło"]),
    ],
)
def test_person_inflections(text, expected):
    assert persons_in(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Kupiłem tusz do drukarki i kubek",  # „tusz” to nie Tusk
        "Na weselu grała duda i skrzypce",  # samo „Duda” nie jest aliasem
        "Pazury kota",  # samo „Pazura” nie jest aliasem
        "Lewandowska",  # Anna, nie Robert
    ],
)
def test_person_false_positives(text):
    assert "Robert Lewandowski" not in persons_in(text)
    assert "Donald Tusk" not in persons_in(text)
    assert "Andrzej Duda" not in persons_in(text)
    assert "Cezary Pazura" not in persons_in(text)


def test_person_tolerates_ocr_noise():
    assert persons_in("", ocr_text="Robert Lewandovski zdradził sekret") == ["Robert Lewandowski"]
    assert persons_in("", ocr_text="Lewandovski") == []  # tolerancja tylko dla aliasów wielowyrazowych


def test_brands_acronyms_are_case_sensitive():
    assert brands_in("Klienci ING i PKO BP oraz ZUS-u") == ["ING", "PKO BP", "ZUS"]
    assert brands_in("investing in banking, zus, pko") == []
    assert brands_in("Akcje Orlenu, gazociąg Baltic Pipe, paczkomaty InPostu") == ["Orlen", "Baltic Pipe", "InPost"]


def test_ambiguous_brand_names_need_unambiguous_form():
    assert brands_in("To fakt, że kampania wyborcza trwa; 10 pudełek") == []
    assert brands_in("Czytaj na fakt.pl i Pudelek.pl") == ["Fakt", "Pudelek"]


def test_canonical_name():
    assert canonical_name("rafal brzoska", "persons") == "Rafał Brzoska"
    assert canonical_name("Robertem Lewandowskim", "persons") == "Robert Lewandowski"
    assert canonical_name("Rafał Brzoska (prezes InPost)", "persons") == "Rafał Brzoska"
    assert canonical_name("PKO Bank Polski", "brands") == "PKO BP"
    assert canonical_name("Narodowy Bank Polski", "brands") == "NBP"
    assert canonical_name("Jan Kowalski", "persons") is None
    assert canonical_name("", "persons") is None


# ------------------------------------------------------------------ narracje


def test_narratives_config_is_valid():
    narratives = load_narratives()
    assert len(narratives) >= 11
    labels = narrative_labels()
    for tag in ("falszywy_wywiad", "inwestycja_krypto", "akcje_spolek", "program_rzadowy", "bot_ai", "alert_bankowy"):
        assert tag in labels and labels[tag]
    assert all(n.keywords and n.patterns and n.min_hits >= 1 for n in narratives)


@pytest.mark.parametrize(
    ("text", "tag"),
    [
        ("Zainwestuj w Bitcoin i kryptowaluty — zysk gwarantowany", "inwestycja_krypto"),
        ("Twoja paczka została wstrzymana. Dopłata do przesyłki 1,99 zł, kurier czeka", "doplata_paczka"),
        ("Wpadka na antenie! Lewandowski ujawnił w wywiadzie swój sekret", "falszywy_wywiad"),
        ("Inwestuj w akcje Orlenu i Baltic Pipe — dywidenda dla każdego", "akcje_spolek"),
        ("Bot AI handluje automatycznie, algorytm sztucznej inteligencji", "bot_ai"),
        ("Twoje konto zostało zablokowane. Podejrzana transakcja — potwierdź dane logowania w banku", "alert_bankowy"),
        ("Nie żyje znany aktor. Cała Polska w szoku, tragedia!", "skandal_celebryty"),
    ],
)
def test_narratives_detected(text, tag):
    assert tag in match_narratives(fold(text))


def test_neutral_text_has_no_narrative():
    text = "Spotkanie klubu książki odbędzie się w czwartek w bibliotece miejskiej. Zapraszamy wszystkich chętnych."
    assert match_narratives(fold(text)) == []


# ------------------------------------------------------------------ obietnica, CTA, produkt, język, słowa kluczowe


def test_promise():
    assert find_promise("Dzięki temu zarabia nawet 5 000 zł dziennie!") == "zarabia nawet 5 000 zł dziennie"
    assert find_promise("Tylko 2500 PLN tygodniowo") == "2500 PLN tygodniowo"
    assert find_promise("Stopa zwrotu aż 300% w miesiąc") == "Stopa zwrotu aż 300%"
    assert find_promise("Earn $1000 a day") == "Earn $1000 a day"
    assert find_promise("Cena: 199 zł, dostawa 2 dni") is None  # cena to nie obietnica


def test_cta_prefers_specific_phrases():
    assert (
        find_cta("Kliknij tutaj. Zarejestruj się teraz i zacznij zarabiać!")
        == "Zarejestruj się teraz i zacznij zarabiać"
    )
    assert find_cta("Zostaw swój numer telefonu, oddzwonimy") == "Zostaw swój numer telefonu, oddzwonimy"
    assert find_cta("zarejestruj sie dzisiaj") == "zarejestruj sie dzisiaj"  # bez polskich znaków
    assert find_cta("Zamówienie zostało przyjęte") is None  # rzeczownik, nie wezwanie


def test_product():
    assert find_product("Zarabia dzięki platformie Quantum AI. Narodowy Bank Polski milczy") == "Quantum AI"
    assert find_product('Aplikacja "Bitcoin Era" działa sama') == "Bitcoin Era"
    assert find_product("Immediate Edge to nowoczesna aplikacja") == "Immediate Edge"
    assert find_product("Wystąpił w programie Dzień Dobry TVN") is None
    assert find_product("PLATFORMA QUANTUM AI ZARABIA ZA CIEBIE") is None  # same wielkie litery
    assert find_product("program Morawieckiego", load_gazetteer("persons")) is None


def test_detect_lang():
    assert detect_lang("To jest reklama, która obiecuje zysk i nie mówi prawdy") == "pl"
    assert detect_lang("this is an advert that promises profit and it is not true") == "en"
    assert detect_lang("Слава Україні героям слава") == "other"
    assert detect_lang("12345 !!!") is None


def test_keywords():
    text = normalize_text(
        "Bitcoin, bitcoina, bitcoinem! Inwestycja w Bitcoin to inwestycja dla każdego. Polityka cookies."
    )
    kws = extract_keywords(text, 3)
    assert kws[0] == "bitcoin" and "inwestycja" in kws
    assert not {"polityka", "cookies", "każdego", "dla"} & set(extract_keywords(text, 20))


# ------------------------------------------------------------------ MinHash


BASE = (
    "Robert Lewandowski ujawnił w wywiadzie swój sposób na pasywny dochód. Dzięki platformie Quantum AI "
    "każdy Polak może zarabiać nawet pięć tysięcy złotych dziennie bez żadnego doświadczenia. Zarejestruj się "
    "dzisiaj, a nasz konsultant zadzwoni i pomoże ci zacząć inwestować w kryptowaluty."
)


def test_minhash_near_duplicates_and_unrelated():
    cfg = TextSettings()
    a = minhash_signature(BASE, cfg)
    variant = minhash_signature(BASE.replace("pięć tysięcy", "sześć tysięcy").replace("dzisiaj", "teraz"), cfg)
    unrelated = minhash_signature(
        "Twoja paczka InPost czeka w paczkomacie na odbiór, prosimy o dopłatę do przesyłki w ciągu doby.", cfg
    )
    assert len(a) == cfg.minhash_perm and all(isinstance(v, int) for v in a)
    assert a == minhash_signature(BASE, cfg)  # deterministyczna
    assert minhash_signature(normalize_text(BASE).upper(), cfg) == a  # niezależna od wielkości liter i formatowania
    assert minhash_jaccard(a, variant) > 0.6
    assert minhash_jaccard(a, unrelated) < 0.1


def test_minhash_short_text_and_helpers():
    assert minhash_signature("tylko pięć słów w tekście", TextSettings()) == []
    assert minhash_signature("", None) == []
    sig_a = minhash_signature(BASE, TextSettings(minhash_perm=64, shingle=2))
    sig_b = minhash_signature(BASE + " Ostatnie słowo.", TextSettings(minhash_perm=64, shingle=2))
    assert len(sig_a) == 64
    assert minhash_object(sig_a).jaccard(minhash_object(sig_b)) == pytest.approx(minhash_jaccard(sig_a, sig_b))
    assert minhash_jaccard([], sig_a) == 0.0 and minhash_jaccard(sig_a, sig_a[:10]) == 0.0


# ------------------------------------------------------------------ text_features (bez LLM)


def test_text_features_full(ctx):
    m = material(
        "SKANDAL NA WIZJI! Robert Lewandowski ujawnił w programie „Dzień Dobry TVN” swój sekret. Dzięki platformie "
        "Quantum AI zarabia nawet 5 000 zł dziennie. Zarejestruj się teraz! Kontakt: +48 600 123 456, "
        "https://quantum-ai-pl.online/start",
        advertiser="Rafał Brzoska Inwestycje",
    )
    tf = text_features(
        m, ["NARODOWY BANK POLSKI chce to zablokować", "Bitcoin rośnie"], "Strona: zysk 300% w skali roku", ctx
    )
    assert tf.input_text.startswith("SKANDAL") and tf.ocr_text.startswith("NARODOWY") and tf.page_text
    assert tf.text == normalize_text(tf.text) and "narodowy bank polski" in tf.text
    assert tf.lang == "pl"
    assert tf.persons == ["Rafał Brzoska", "Robert Lewandowski"]  # reklamodawca też jest analizowany
    assert {"TVN", "NBP", "Bitcoin"} <= set(tf.brands)  # NBP z tekstu OCR
    assert "falszywy_wywiad" in tf.narratives
    assert tf.product == "Quantum AI"
    assert tf.promise == "zarabia nawet 5 000 zł dziennie"
    assert tf.cta == "Zarejestruj się teraz"
    assert tf.iocs["phone"] == ["+48600123456"]
    assert tf.iocs["domain"] == ["quantum-ai-pl.online"]
    assert len(tf.minhash) == ctx.settings.text.minhash_perm
    assert 0 < len(tf.keywords) <= ctx.settings.text.keywords
    assert tf.llm_used is False


def test_text_features_empty_and_partial_inputs(ctx):
    empty = text_features(material(None), [], "", ctx)
    assert empty.text == "" and empty.minhash == [] and empty.persons == [] and empty.lang is None
    assert empty.iocs == {}
    only_page = text_features(material(None), None, "Twoja paczka została wstrzymana, dopłata do przesyłki 2 zł.", ctx)
    assert only_page.page_text and "doplata_paczka" in only_page.narratives
    garbage = text_features(material("\x00\ufeff???"), ["", "   "], None, ctx)
    assert garbage.ocr_text == "" and garbage.minhash == []


def test_text_features_caps_long_inputs(ctx):
    long_page = "reklama inwestycji " * 5000
    tf = text_features(material("krótki tekst reklamy"), [], long_page, ctx)
    assert len(tf.page_text) <= 20_000 and len(tf.text) <= 20_000
    assert tf.text.startswith("krótki tekst reklamy")


def test_text_features_survives_broken_step(ctx, monkeypatch):
    import slad.features.text as text_module

    def boom(*_a, **_k):
        raise RuntimeError("awaria")

    monkeypatch.setattr(text_module, "find_promise", boom)
    tf = text_features(material("Robert Lewandowski zarabia 5000 zł dziennie, zadzwoń 600 123 456"), [], "", ctx)
    assert tf.promise is None
    assert tf.persons == ["Robert Lewandowski"] and tf.iocs["phone"] == ["+48600123456"]


# ------------------------------------------------------------------ OCR


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # starszy Pillow
        return ImageFont.load_default()


def _text_image(path: Path, lines: list[str], size=(1100, 340)) -> Path:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((40, 40 + i * 140), line, fill="black", font=_font(72))
    img.save(path)
    return path


class FakeEngine:
    """Udaje RapidOCR: zwraca zadane segmenty (czworokąty + teksty)."""

    def __init__(self, items):
        self.items = items
        self.calls = 0

    def __call__(self, img):
        self.calls += 1
        boxes = [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]] for (x0, y0, x1, y1), _ in self.items]
        return SimpleNamespace(boxes=boxes, txts=tuple(t for _, t in self.items))


def test_group_lines_reading_order():
    segments = [
        (300, 12, 500, 42, "świat"),
        (10, 10, 280, 40, "Witaj"),
        (10, 60, 200, 90, "druga linia"),
        (220, 64, 400, 92, "ciąg dalszy"),
    ]
    assert ocr.group_lines(segments) == ["Witaj świat", "druga linia ciąg dalszy"]
    assert ocr.group_lines([]) == []


def test_tiles_downscale_and_tall_page_cuts():
    wide = Image.new("RGB", (4000, 1000), "white")
    (single,) = ocr.tiles(wide, 2000)
    assert single.size == (2000, 500)
    tall = Image.new("RGB", (600, 5000), "white")
    draw = ImageDraw.Draw(tall)
    for y in range(0, 5000, 50):  # „linie tekstu” co 50 px, odstępy między nimi puste
        draw.rectangle((20, y + 10, 580, y + 30), fill="black")
    parts = ocr.tiles(tall, 2000)
    assert len(parts) >= 3 and sum(p.size[1] for p in parts) == 5000
    assert all(max(p.size) <= 2000 for p in parts)
    for p in parts[:-1]:  # cięcie w pustym wierszu — nie przez „linię tekstu”
        last_row = p.crop((0, p.size[1] - 1, p.size[0], p.size[1])).convert("L")
        assert last_row.getextrema() == (255, 255)


def test_ocr_with_fake_engine_and_cache(ctx, tmp_path, monkeypatch):
    path = _text_image(tmp_path / "ad.png", ["X"])
    engine = FakeEngine([((10, 60, 200, 90), "dolna"), ((10, 10, 100, 40), "górna"), ((120, 12, 300, 42), "linia")])
    monkeypatch.setattr(ocr, "_get_engine", lambda: engine)
    assert ocr.ocr_image(path, ctx) == "górna linia\ndolna"
    assert ocr.ocr_image(path, ctx) == "górna linia\ndolna"
    assert engine.calls == 1  # drugi raz z cache (klucz = sha256 pliku)
    assert ocr.file_sha256(path) in ctx._caches["ocr"]


def test_ocr_unavailable_engine_returns_empty_and_does_not_cache(ctx, tmp_path, monkeypatch):
    path = _text_image(tmp_path / "ad.png", ["X"])
    monkeypatch.setattr(ocr, "_get_engine", lambda: None)
    assert ocr.ocr_image(path, ctx) == ""
    assert not ctx._caches["ocr"]


def test_ocr_engine_failure_returns_empty(ctx, tmp_path, monkeypatch):
    path = _text_image(tmp_path / "ad.png", ["X"])

    def broken(_img):
        raise RuntimeError("onnxruntime error")

    monkeypatch.setattr(ocr, "_get_engine", lambda: broken)
    assert ocr.ocr_image(path, ctx) == ""


def test_ocr_unreadable_file_and_disabled(ctx, tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"to nie jest obraz")
    with pytest.raises(ValueError):
        ocr.ocr_image(bad, ctx)
    with pytest.raises(ValueError):
        ocr.ocr_image(tmp_path / "missing.png", ctx)
    ctx.settings.ocr.enabled = False
    try:
        assert ocr.ocr_image(bad, ctx) == ""
    finally:
        ctx.settings.ocr.enabled = True


def test_load_image_handles_transparency(tmp_path):
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (40, 20), (0, 0, 0, 0)).save(path)
    img = ocr.load_image(path)
    assert img.mode == "RGB" and img.getpixel((5, 5)) == (255, 255, 255)


def test_ocr_real_engine(ctx, tmp_path):
    path = _text_image(tmp_path / "real.png", ["PLATFORMA INWESTYCYJNA", "ZAROBISZ 5000 PLN"])
    text = ocr.ocr_image(path, ctx)
    if not text:
        pytest.skip("RapidOCR niedostępny w tym środowisku")
    lines = text.upper().splitlines()
    assert any("PLATFORMA" in ln for ln in lines) and any("5000" in ln for ln in lines)
    assert next(i for i, ln in enumerate(lines) if "PLATFORMA" in ln) < next(
        i for i, ln in enumerate(lines) if "5000" in ln
    )
