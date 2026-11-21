"""Cechy tekstowe materiału: tekst wejściowy + OCR obrazów + tekst strony docelowej →
osoby, marki, narracje, produkt, obietnica, wezwanie do działania (CTA), IOC, słowa kluczowe, język, MinHash.

Publiczne API (kontrakt: docs/ARCHITEKTURA.md):
    text_features(material, ocr_texts, page_text, ctx) -> TextFeatures
    minhash_signature(text, cfg) -> list[int]   # datasketch MinHash: scheme="affine32", seed=1, szyngle słów
    normalize_text(text) -> str                 # NFKC, małe litery, jednolite cudzysłowy i myślniki, bez znaków
                                                #   zerowej szerokości i homoglifów (cyrylica w łacińskich słowach)
    fold(text) -> str                           # normalize_text bez polskich znaków (do dopasowań)
Pomocnicze: minhash_jaccard(a, b), minhash_object(sig) — dla LSH (datasketch >= 2.0 wymaga podania scheme),
load_gazetteer(kind), canonical_name(name, kind), load_narratives(), narrative_labels(), detect_lang(text),
extract_keywords(text, n), find_promise / find_cta / find_product(text).

Bez LLM działają heurystyki: słowniki config/gazetteer/{persons,brands}.yaml, reguły config/narratives.yaml
i wyrażenia regularne. Z LLM (ctx.llm) wyniki są łączone: osoby i marki sprowadzane do nazw ze słownika,
narracje tylko z dozwolonych tagów, produkt / obietnica / CTA uzupełniane, gdy heurystyki nic nie znalazły
(TextFeatures.llm_used = True). Błąd jednej cechy nie przerywa pozostałych: puste pole + ostrzeżenie w logu.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from rapidfuzz import fuzz

from slad.features.iocs import extract_iocs
from slad.models import Material, TextFeatures
from slad.settings import CONFIG_DIR, TextSettings

log = logging.getLogger(__name__)

GAZETTEER_DIR = CONFIG_DIR / "gazetteer"
NARRATIVES_PATH = CONFIG_DIR / "narratives.yaml"

MAX_TEXT = 20_000  # TextFeatures.text — złączony tekst do porównań i MinHash
MAX_STORED = 20_000  # input_text / ocr_text / page_text zapisywane w FeatureSet
MAX_INPUT_PART = 8_000  # udział tekstu wejściowego i OCR w złączonym tekście (resztę wypełnia strona docelowa)
MAX_NARRATIVES = 4  # z heurystyk; LLM może dodać jeszcze jedną
MAX_LLM_EXTRA = 5  # ile osób/marek spoza słownika może dodać LLM
MIN_TOKENS = 6  # krótszy tekst → brak sygnatury MinHash
MINHASH_SEED = 1
MINHASH_SCHEME = "affine32"  # datasketch >= 2.0: przy odtwarzaniu MinHash z hashvalues trzeba podać scheme
FUZZY_MIN_SCORE = 90  # rapidfuzz.partial_ratio — tolerancja błędów OCR w aliasach wielowyrazowych
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)  # libyaml, gdy dostępna (szybsze wczytywanie)

# ------------------------------------------------------------------ normalizacja

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u00ad\u180e\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\ufeff"))
_PUNCT = str.maketrans(
    {
        **dict.fromkeys("„”“‟«»❝❞〝〞″", '"'),
        **dict.fromkeys("‘’‚‛‹›′´`", "'"),
        **dict.fromkeys("‐‑‒–—―−⸺⸻﹘﹣", "-"),
    }
)
# Homoglify: cyrylica / greka udające łacinę w słowach łacińskich („Bіtcoin” z ukraińskim „і”) — typowy sposób
# omijania filtrów reklam. Zamieniane tylko w słowach mieszanych, więc prawdziwy tekst cyrylicą zostaje.
_HOMOGLYPHS = str.maketrans(
    {
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
        "ԁ": "d", "һ": "h", "ӏ": "l", "ԛ": "q", "ԝ": "w", "ɡ": "g",
        "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
        "Х": "X", "У": "Y", "І": "I", "Ј": "J", "Ѕ": "S",
        "α": "a", "ο": "o", "ν": "v", "ρ": "p", "τ": "t", "ι": "i", "κ": "k", "υ": "u",
        "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
        "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    }
)  # fmt: skip
_HOMOGLYPH_CHARS = frozenset(chr(k) for k in _HOMOGLYPHS)
_WORD = re.compile(r"\w+")
_FOLD_SPECIAL = str.maketrans({"ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ø": "o", "Ø": "O", "ı": "i"})


def _fix_homoglyphs(text: str) -> str:
    if _HOMOGLYPH_CHARS.isdisjoint(text):
        return text

    def repl(m: re.Match[str]) -> str:
        w = m.group(0)
        if any(c in _HOMOGLYPH_CHARS for c in w) and any(c.isascii() and c.isalpha() for c in w):
            return w.translate(_HOMOGLYPHS)
        return w

    return _WORD.sub(repl, text)


def _base_clean(text: str) -> str:
    return _fix_homoglyphs(unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).translate(_PUNCT))


def normalize_text(text: str | None) -> str:
    """NFKC, bez znaków zerowej szerokości i homoglifów, jednolite cudzysłowy/myślniki, małe litery, pojedyncze
    spacje. Polskie znaki zostają."""
    if not text:
        return ""
    return " ".join(_base_clean(text).lower().split())


def strip_diacritics(text: str) -> str:
    """Usuwa znaki diakrytyczne (ą→a, ł→l, ż→z…) bez innych zmian — bezpieczne także dla wyrażeń regularnych."""
    decomposed = unicodedata.normalize("NFD", text.translate(_FOLD_SPECIAL))
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if not unicodedata.combining(c)))


def fold(text: str | None) -> str:
    """Tekst do dopasowań: normalize_text + bez polskich znaków („Rafał Brzoską” → „rafal brzoska”)."""
    return strip_diacritics(normalize_text(text))


def _clean_lines(text: str) -> str:
    """Tekst do wycinania fraz (obietnica, CTA, produkt): wielkość liter i podział na linie zachowane."""
    lines = (" ".join(line.split()) for line in _base_clean(text).splitlines())
    return "\n".join(line for line in lines if line)


def _cap(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]


_PL_CLASSES = {"a": "[aą]", "c": "[cć]", "e": "[eę]", "l": "[lł]", "n": "[nń]", "o": "[oó]", "s": "[sś]", "z": "[zźż]"}


def _pl(pattern: str) -> str:
    r"""Wzorzec zapisany bez polskich znaków → toleruje je (a→[aą], z→[zźż]…); spacja → \s+."""
    out, i = [], 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            out.append(pattern[i : i + 2])
            i += 2
            continue
        out.append(r"\s+" if ch == " " else _PL_CLASSES.get(ch, ch))
        i += 1
    return "".join(out)


def _safe[T](label: str, default: T, fn: Callable[..., T], *args: Any) -> T:
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001 — jedna cecha nie może zatrzymać pozostałych
        log.warning("Cechy tekstowe: błąd w kroku „%s” — pole pozostaje puste", label, exc_info=True)
        return default


# ------------------------------------------------------------------ słowniki osób i marek (gazetteer)

_ENDINGS_CONSONANT = "a|u|e|y|i|em|ie|iem|owi|om|ach|ami|ow|owie|owa|owej"
_ENDINGS_VOWEL = "a|e|i|y|u|o|ie|ej|em|om|ach|ami|owi|ego|emu|ym|ze|zie"
_ALIAS_SEP = r"[\s.\-]+"


def _inflected(token: str) -> str:
    """Regex tokenu aliasu (tekst złożony, bez polskich znaków) z typowymi polskimi końcówkami fleksyjnymi."""
    if not token.isalpha() or len(token) < 3:
        return re.escape(token)
    if token.endswith(("ski", "cki", "dzki")):  # przymiotnikowe męskie: Lewandowski, Lewandowskiego, -skim
        return re.escape(token[:-1]) + "(?:i|iego|iemu|im)"
    if token.endswith(("ska", "cka", "dzka")) and len(token) >= 6:  # żeńskie: Lewandowska, Lewandowskiej
        return re.escape(token[:-1]) + "(?:a|iej)"
    if token[-1] in "aeiouy":  # Duda → Dudy, Dudzie; Hołownia → Hołowni
        return re.escape(token[:-1]) + f"(?:{_ENDINGS_VOWEL})"
    return re.escape(token) + f"(?:{_ENDINGS_CONSONANT})?"  # Tusk → Tuska, Tuskiem; Orlen → Orlenu


def _is_acronym(alias: str) -> bool:
    """Skrót z 2–3 wielkich liter (PKO, ING, ZUS) — dopasowywany z rozróżnieniem wielkości liter."""
    letters = alias.replace(" ", "")
    return 2 <= len(letters) <= 3 and letters.isascii() and letters.isalpha() and letters.isupper()


@dataclass(frozen=True)
class GazEntry:
    name: str  # nazwa kanoniczna (wynik)
    patterns: tuple[tuple[str, re.Pattern[str]], ...]  # (fragment do szybkiego filtra, regex) na tekście złożonym
    cased: tuple[re.Pattern[str], ...]  # skróty WIELKIMI literami — na tekście z zachowaną wielkością liter
    fuzzy: tuple[str, ...]  # złożone aliasy wielowyrazowe — tolerancja błędów OCR
    keys: frozenset[str]  # wszystkie złożone zapisy — kanonizacja wyników LLM


def _entry(item: dict[str, Any]) -> GazEntry | None:
    name = " ".join(unicodedata.normalize("NFKC", str(item.get("name") or "")).split())
    if not name:
        return None
    aliases = [str(a) for a in item.get("aliases") or [] if str(a).strip()]
    if item.get("match_name", True):
        aliases.insert(0, name)
    patterns: dict[str, tuple[str, re.Pattern[str]]] = {}
    cased: dict[str, re.Pattern[str]] = {}
    fuzzy: list[str] = []
    keys = {fold(name)}
    for alias in aliases:
        alias = " ".join(unicodedata.normalize("NFKC", alias).split())
        folded = fold(alias)
        if not folded:
            continue
        keys.add(folded)
        if _is_acronym(alias):
            body = r"\s+".join(map(re.escape, alias.split()))
            cased.setdefault(alias, re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])"))
            continue
        if folded in patterns:
            continue
        tokens = folded.split()
        longest = max(tokens, key=len)
        needle = longest[:-1] if longest.isalpha() and len(longest) >= 3 else longest
        body = _ALIAS_SEP.join(_inflected(t) for t in tokens)
        patterns[folded] = (needle, re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])"))
        if len(tokens) >= 2 and len(folded) >= 8:
            fuzzy.append(folded)
    return GazEntry(name, tuple(patterns.values()), tuple(cased.values()), tuple(fuzzy), frozenset(keys))


@lru_cache(maxsize=8)
def load_gazetteer(kind: str) -> tuple[GazEntry, ...]:
    """config/gazetteer/<kind>.yaml (persons | brands): lista {name, aliases[], match_name?}."""
    path = GAZETTEER_DIR / f"{kind}.yaml"
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_YAML_LOADER) or []
    except (OSError, yaml.YAMLError) as e:
        log.warning("Słownik %s niedostępny (%s) — pomijam", path, e)
        return ()
    if isinstance(data, dict):
        data = data.get("entries") or data.get(kind) or []
    entries: list[GazEntry] = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, str):
            item = {"name": item}
        if isinstance(item, dict):
            entry = _entry(item)
            if entry is not None:
                entries.append(entry)
    return tuple(entries)


def _fuzzy_in(alias: str, text: str) -> bool:
    """Cały alias występuje w tekście z drobnymi błędami (partial_ratio dopasowuje KRÓTSZY napis do dłuższego —
    gdy tekst jest krótszy od aliasu, porównujemy całość, żeby samo „Lewandovski” nie dało „Robert Lewandowski”)."""
    if len(text) < len(alias):
        return fuzz.ratio(alias, text, score_cutoff=FUZZY_MIN_SCORE) > 0
    return fuzz.partial_ratio(alias, text, score_cutoff=FUZZY_MIN_SCORE) > 0


def match_gazetteer(entries: Sequence[GazEntry], folded: str, cased_text: str = "", ocr_folded: str = "") -> list[str]:
    """Nazwy kanoniczne wpisów znalezionych w tekście, w kolejności pierwszego wystąpienia.

    folded: tekst złożony (fold); cased_text: ten sam tekst z wielkością liter (dla skrótów);
    ocr_folded: złożony tekst z OCR — dla aliasów wielowyrazowych z drobnymi błędami odczytu (partial_ratio >= 90).
    """
    found: list[tuple[int, str]] = []
    for idx, entry in enumerate(entries):
        pos: int | None = None
        for needle, rx in entry.patterns:
            if needle in folded:
                m = rx.search(folded)
                if m and (pos is None or m.start() < pos):
                    pos = m.start()
        for rx in entry.cased:
            m = rx.search(cased_text)
            if m and (pos is None or m.start() < pos):
                pos = m.start()
        if pos is None and ocr_folded and entry.fuzzy and any(_fuzzy_in(a, ocr_folded) for a in entry.fuzzy):
            pos = len(folded) + idx
        if pos is not None:
            found.append((pos, entry.name))
    return [name for _, name in sorted(found)]


def canonical_name(name: str, entries: Sequence[GazEntry] | str) -> str | None:
    """Nazwa ze słownika dla dowolnego zapisu („rafal brzoska”, „Brzoską”, „PKO Bank Polski”) albo None."""
    db = load_gazetteer(entries) if isinstance(entries, str) else entries
    f = fold(name)
    if not f:
        return None
    for e in db:
        if f in e.keys:
            return e.name
    for e in db:
        if any(rx.search(f) for _, rx in e.patterns):
            return e.name
    for e in db:
        if any(" " in k and len(k) >= 8 and fuzz.ratio(f, k, score_cutoff=FUZZY_MIN_SCORE) for k in e.keys):
            return e.name
    return None


# ------------------------------------------------------------------ narracje

_REGEX_META = frozenset(".^$*+?{}[]\\|()")


@dataclass(frozen=True)
class Narrative:
    tag: str
    label: str
    keywords: tuple[tuple[str | None, re.Pattern[str]], ...]  # (dosłowny fragment do filtra lub None, regex)
    patterns: tuple[re.Pattern[str], ...]
    min_hits: int


@lru_cache(maxsize=4)
def load_narratives(path: str | None = None) -> tuple[Narrative, ...]:
    """config/narratives.yaml → reguły narracji (opis formatu w nagłówku pliku)."""
    file = Path(path) if path else NARRATIVES_PATH
    try:
        data = yaml.load(file.read_text(encoding="utf-8"), Loader=_YAML_LOADER) or {}
    except (OSError, yaml.YAMLError) as e:
        log.warning("Reguły narracji %s niedostępne (%s) — pomijam", file, e)
        return ()
    out: list[Narrative] = []
    for tag, spec in data.items() if isinstance(data, dict) else []:
        spec = spec if isinstance(spec, dict) else {}
        keywords: list[tuple[str | None, re.Pattern[str]]] = []
        for kw in spec.get("keywords") or []:
            k = strip_diacritics(str(kw)).strip()
            if not k:
                continue
            try:
                rx = re.compile(rf"(?<![a-z0-9])(?:{k})", re.I)
            except re.error as e:
                log.warning("narratives.yaml, %s: błędne słowo kluczowe %r (%s)", tag, kw, e)
                continue
            keywords.append((None if _REGEX_META & set(k) else k.lower(), rx))
        patterns: list[re.Pattern[str]] = []
        for p in spec.get("patterns") or []:
            try:
                patterns.append(re.compile(strip_diacritics(str(p)), re.I))
            except re.error as e:
                log.warning("narratives.yaml, %s: błędny wzorzec %r (%s)", tag, p, e)
        label = str(spec.get("label_pl") or tag)
        min_hits = int(spec.get("min_hits") or 2)
        out.append(Narrative(str(tag), label, tuple(keywords), tuple(patterns), min_hits))
    return tuple(out)


def narrative_labels() -> dict[str, str]:
    """Dozwolone tagi narracji → opis po polsku (UI, prompt LLM)."""
    return {n.tag: n.label for n in load_narratives()}


def match_narratives(folded: str, narratives: Sequence[Narrative] | None = None) -> list[str]:
    """Tagi narracji (najsilniejsze pierwsze, maks. MAX_NARRATIVES): słowo kluczowe = 1 pkt, wzorzec = 2 pkt."""
    scored: list[tuple[int, int, str]] = []
    for order, n in enumerate(load_narratives() if narratives is None else narratives):
        hits = sum(1 for needle, rx in n.keywords if (needle is None or needle in folded) and rx.search(folded))
        hits += 2 * sum(1 for rx in n.patterns if rx.search(folded))
        if hits >= n.min_hits:
            scored.append((-hits, order, n.tag))
    return [tag for _, _, tag in sorted(scored)][:MAX_NARRATIVES]


# ------------------------------------------------------------------ język i słowa kluczowe

_PL_STOP_BASE = """
a aby ach acz aczkolwiek aj albo ale ależ ani aż bardziej bardzo bez bo bowiem by byli bym byś być był była było były
będzie będą będę będziesz cali cała cały cię ciebie co cokolwiek coraz coś czasami czasem czemu czy czyli często daleko
dla dlaczego dlatego do dobrze dokąd dość dużo dwa dwaj dwie dwoje dzisiaj dziś gdy gdyby gdyż gdzie gdziekolwiek gdzieś
go i ich ile im inna inne inny innych iż ja jak jakaś jakby jaki jakichś jakie jakiś jakiż jakkolwiek jako jakoś je
jeden jedna jedno jednak jednakże jego jej jemu jest jestem jesteś jeszcze jeśli jeżeli już ją każdy każda każde kiedy
kilka kimś kto ktokolwiek ktoś która które którego której który których którym którzy ku lecz lub ma mają mam mamy mało
mi między mną mnie mogą moi moim moja moje może można mój mu musi my na nad nam nami nas nasi nasz nasza nasze naszego
naszej naszych natomiast nawet nic nich nie niech niego niej niemu nigdy nim nimi niż no o obok od około on ona one oni
ono oraz oto owszem pan pana pani panie po pod podczas pomimo ponad ponieważ powinien powinna powinni powinno poza prawie
przecież przed przede przez przy również sam sama są się skąd sobie sobą sposób swoje swój swoich swoją swojego swojej ta
tak taka taki takie także tam te tego tej temu ten teraz też to tobie toteż trzeba tu tutaj twoi twoim twoja twoje twój
ty tych tylko tym u w wam wami was wasz wasza wasze we według wiele wielu więc więcej wszyscy wszystkich wszystkie
wszystkim wszystko wtedy wy właśnie z za zapewne zawsze ze znowu znów został została zostały zostać żaden żadna żadne
żadnych że żeby każdego każdej każdemu każdym każdą dzięki czego czym nią którą któremu swoim swoją naszym naszą
waszym zatem jutro około ponad lecz bardzo także oraz
""".split()
_EN_STOP = frozenset(
    """
a about above after again against all also am an and any are as at be because been before being below between both but
by can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other
our ours ourselves out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves get got one may might must shall
""".split()
)
_PL_STOP = frozenset(_PL_STOP_BASE) | frozenset(strip_diacritics(w) for w in _PL_STOP_BASE)
_BOILERPLATE = frozenset(
    strip_diacritics(w)
    for w in """
cookie cookies polityka polityki prywatności regulamin regulaminu copyright rights reserved privacy policy terms home
menu login zaloguj kontakt kontaktu strona strony stronie czytaj komentarze komentarz komentarzy udostępnij lubię
sponsorowane sponsored like share comment reply odpowiedz godz godzin godziny minut temu wczoraj https http html
zobacz kliknij tutaj więcej
""".split()
)
_KEYWORD_STOP = _PL_STOP | _EN_STOP | _BOILERPLATE
_LETTER_WORD = re.compile(r"[^\W\d_]+")
_PL_LETTERS = re.compile(r"[ąćęłńóśźż]")


def detect_lang(text: str) -> str | None:
    """„pl” / „en” / „other” (słowa funkcyjne + polskie znaki); None dla tekstu bez słów."""
    words = _LETTER_WORD.findall(text.lower())
    if not words:
        return None
    pl = sum(w in _PL_STOP for w in words) + 0.5 * min(5, sum(1 for w in words if _PL_LETTERS.search(w)))
    en = sum(w in _EN_STOP for w in words)
    if max(pl, en) < max(1.0, 0.03 * len(words)):
        return "other"
    return "pl" if pl >= en else "en"


def extract_keywords(text: str, n: int = 12) -> list[str]:
    """Najczęstsze słowa (>= 4 litery, bez słów funkcyjnych PL/EN i elementów strony). Odmiany jednego słowa
    liczone razem (wspólny 5-literowy początek), zwracana najczęstsza forma."""
    counts: Counter[str] = Counter()
    first: dict[str, int] = {}
    forms: dict[str, Counter[str]] = {}
    for i, word in enumerate(_LETTER_WORD.findall(text.lower())):
        if len(word) < 4:
            continue
        f = strip_diacritics(word)
        if f in _KEYWORD_STOP:
            continue
        key = f[:5] if len(f) >= 6 else f
        counts[key] += 1
        first.setdefault(key, i)
        forms.setdefault(key, Counter())[word] += 1
    ranked = sorted(counts, key=lambda k: (-counts[k], first[k]))
    return [forms[k].most_common(1)[0][0] for k in ranked[: max(0, n)]]


# ------------------------------------------------------------------ obietnica, CTA, produkt

_CUR = r"(?:zł|zl\b|pln|złotych|zlotych|złote|zlote|euro|eur\b|€|\$|usd|dolar\w*)"
_NUM = r"\d{1,3}(?:[ \u00a0.,]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d+)?"
_AMOUNT = rf"(?:(?:{_NUM})\s?(?:tys\.?|tysięcy|tysiecy|mln|k)?\s?{_CUR}|[$€]\s?(?:{_NUM})(?:\s?(?:tys\.?|mln|k))?)"
_PERIOD = (
    r"(?:dziennie|codziennie|tygodniowo|miesięcznie|miesiecznie|rocznie|"
    r"na\s+(?:dzień|dzien|tydzień|tydzien|miesiąc|miesiac|godzinę|godzine|rok)|"
    r"w\s+(?:tydzień|tydzien|miesiąc|miesiac|dzień|dzien|rok)|"
    r"w\s+(?:ciągu|ciagu)\s+(?:\d+\s+)?\w+|"
    r"/\s?(?:dzień|dzien|dobę|dobe|tydz\w*|mies\w*|mc|h|godz\w*)|"
    r"(?:per|a)\s+(?:day|week|month)|daily|weekly|monthly)"
)
_GAIN = (
    r"(?:zarob\w*|zarabia\w*|zysk\w*|dochod\w*|dochód\w*|przych\w*|wypłat\w*|wyplat\w*|wypłac\w*|wyplac\w*|"
    r"pasywn\w*|otrzym\w*|zgarn\w*|pomnóż\w*|pomnoz\w*|earn\w*|profit\w*|income)"
)
_PROMISE_RULES = (  # kolejność = priorytet
    re.compile(rf"{_GAIN}[^.!?\n]{{0,60}}?{_AMOUNT}(?:\s*{_PERIOD})?", re.I),
    re.compile(rf"{_AMOUNT}\s*{_PERIOD}", re.I),
    re.compile(rf"(?:{_GAIN}|zwrot\w*|stop\w+\s+zwrotu|rentown\w*|\broi\b)[^.!?\n]{{0,40}}?\d+(?:[.,]\d+)?\s?%", re.I),
    re.compile(
        r"\d+(?:[.,]\d+)?\s?%\s*(?:zysku|zwrotu|dziennie|miesięcznie|miesiecznie|rocznie|w\s+skali\s+\w+|profit|return)",
        re.I,
    ),
    re.compile(r"(?:-\s?)?\d{2}\s?%\s*(?:taniej|rabatu|zniżki|znizki|off)\b", re.I),
)

# Wezwania do działania — od najbardziej konkretnych (rejestracja, zostawienie numeru) do ogólnych („sprawdź”)
CTA_PHRASES = (
    "zarejestruj sie",
    "zaloz (?:darmowe )?konto",
    "zostaw (?:swoj |swoje )?(?:numer|telefon|dane|kontakt)",
    "wypelnij (?:formularz|ankiete)",
    "zainwestuj",
    "zacznij (?:zarabiac|inwestowac|teraz|juz)",
    "odbierz",
    "zamow",
    "wplac",
    "oplac",
    "zaplac",
    "zaktualizuj",
    "potwierdz",
    "uzupelnij",
    "zweryfikuj",
    "kup (?:teraz|juz)",
    "dolacz",
    "zapisz sie",
    "pobierz",
    "zadzwon",
    "napisz",
    "skorzystaj",
    "sprawdz",
    "kliknij",
    "dowiedz sie wiecej",
    "wejdz",
    "przejdz",
    "register",
    "sign up",
    "join (?:now|us|today)",
    "claim",
    "buy now",
    "order now",
    "get started",
    "click here",
    "learn more",
)
_CTA_RULES = tuple(re.compile(rf"(?<!\w){_pl(p)}(?!\w)[^.!?\n|•]{{0,80}}", re.I) for p in CTA_PHRASES)

_PRODUCT_NOUN = (  # mocne: platforma / aplikacja / bot…; słabe (program, serwis, portal, projekt) — w drugiej kolejności
    r"(?:platform\w*|aplikacj\w*|bot(?:a|em|y)?|robot\w*|system\w*|algorytm\w*|narzędzi\w*|narzedzi\w*|giełd\w*|"
    r"gield\w*|app|program(?!ie\b)\w*|serwis\w*|portal\w*|projekt\w*)"
)
_WEAK_PRODUCT_NOUN = re.compile(r"(?:program|serwis|portal|projekt)", re.I)
_PRODUCT_ADJ = (
    r"(?:(?:inwestycyjn|tradingow|handlow|automatyczn|nowoczesn|innowacyjn|kryptowalutow|finansow|internetow|nazwan|"
    r"zwan|rewolucyjn|now)\w*\s+|o\s+nazwie\s+){0,3}"
)
_UPPER = "A-ZĄĆĘŁŃÓŚŹŻ"
_NAME_TAIL = r"[\w+&-]*(?:\.[\w+&-]+)*"  # kropka tylko wewnątrz nazwy (Trade.io), nie na końcu zdania
_NAME_WORD = rf"[{_UPPER}0-9]{_NAME_TAIL}"
_NAME = rf"[{_UPPER}]{_NAME_TAIL}(?:\s+(?:{_NAME_WORD}|AI|X|Pro|Plus|GPT|Max)){{0,3}}"
_PRODUCT_FWD = re.compile(rf"(?<!\w)((?i:{_PRODUCT_NOUN}))\s+(?i:{_PRODUCT_ADJ})([\"'„“«]?)({_NAME})")
_PRODUCT_REV = re.compile(rf"(?<![\w.])({_NAME})\s+(?i:to|jest|is)\s+(?:\w+\s+){{0,2}}?((?i:{_PRODUCT_NOUN}))(?!\w)")
_PRODUCT_STOP = _PL_STOP | _EN_STOP | frozenset("polska polski polsce polakow polacy nowy nowa nowe".split())


def _trim(text: str, limit: int) -> str:
    text = " ".join(text.split()).strip(" ,;:-–\"'")
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-") + "…"
    return text


def find_promise(text: str) -> str | None:
    """Obiecywana korzyść: kwota z walutą i okresem („5000 zł dziennie”), „zarobisz…”, procent zysku, rabat."""
    for rx in _PROMISE_RULES:
        m = rx.search(text)
        if m:
            return _trim(m.group(0), 120) or None
    return None


def find_cta(text: str) -> str | None:
    """Wezwanie do działania (najbardziej konkretne z listy CTA_PHRASES) wraz z resztą zdania."""
    for rx in _CTA_RULES:
        m = rx.search(text)
        if m:
            return _trim(m.group(0), 100) or None
    return None


def find_product(text: str, persons: Sequence[GazEntry] = ()) -> str | None:
    """Nazwa „produktu” — wielką literą przy słowach platforma / aplikacja / bot / system („platforma Quantum AI”,
    „Immediate Edge to aplikacja…”). Pomija nazwiska ze słownika osób i tekst pisany w całości wielkimi literami."""
    candidates: list[tuple[bool, int, str]] = []
    for m in _PRODUCT_FWD.finditer(text):
        noun, quoted, name = m.group(1), m.group(2), m.group(3)
        if noun.isupper() and name.isupper() and not quoted:
            continue
        candidates.append((bool(_WEAK_PRODUCT_NOUN.match(noun)), m.start(3), name))
    for m in _PRODUCT_REV.finditer(text):
        name, noun = m.group(1), m.group(2)
        if noun.isupper() and name.isupper():
            continue
        candidates.append((bool(_WEAK_PRODUCT_NOUN.match(noun)), m.start(1), name))
    for _, _, name in sorted(candidates):
        name = name.rstrip(".-+&")
        if len(name) < 3 or fold(name.split()[0]) in _PRODUCT_STOP:
            continue
        if persons and match_gazetteer(persons, fold(name)):
            continue
        return _trim(name, 50)
    return None


# ------------------------------------------------------------------ MinHash

_TOKEN = re.compile(r"[^\W_]+")


def minhash_signature(text: str, cfg: TextSettings | None = None) -> list[int]:
    """Sygnatura MinHash (datasketch, scheme="affine32", seed=1) szyngli cfg.shingle słów tekstu złożonego.
    Pusta lista, gdy tekst ma mniej niż MIN_TOKENS słów."""
    tokens = _TOKEN.findall(fold(text))
    if len(tokens) < MIN_TOKENS:
        return []
    from datasketch import MinHash

    k = max(1, min(int(getattr(cfg, "shingle", 3) or 3), len(tokens)))
    num_perm = int(getattr(cfg, "minhash_perm", 128) or 128)
    shingles = {" ".join(tokens[i : i + k]).encode("utf-8") for i in range(len(tokens) - k + 1)}
    m = MinHash(num_perm=num_perm, seed=MINHASH_SEED, scheme=MINHASH_SCHEME)
    m.update_batch(shingles)
    return [int(v) for v in m.hashvalues]


def minhash_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    """Szacowany współczynnik Jaccarda dwóch sygnatur (0 dla pustych lub różnej długości)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def minhash_object(signature: Sequence[int]) -> Any:
    """datasketch.MinHash z zapisanej sygnatury (np. do MinHashLSH)."""
    from datasketch import MinHash

    return MinHash(hashvalues=list(signature), seed=MINHASH_SEED, scheme=MINHASH_SCHEME)


# ------------------------------------------------------------------ LLM


def _llm_input(input_text: str, ocr_text: str, page_text: str) -> str:
    parts = []
    if input_text:
        parts.append("[Tekst reklamy / opis materiału]\n" + input_text)
    if ocr_text:
        parts.append("[Tekst odczytany z obrazu (OCR, możliwe błędy)]\n" + ocr_text)
    if page_text:
        parts.append("[Tekst strony docelowej]\n" + page_text)
    return "\n\n".join(parts)


def _clean_name(raw: str) -> str | None:
    name = " ".join(str(raw).split()).strip(" \"'.,;:()[]")
    if len(name) < 2 or len(name) > 80:
        return None
    return name.title() if name.islower() else name


def _merge_names(current: list[str], extra: Any, entries: Sequence[GazEntry]) -> list[str]:
    out = list(current)
    seen = {fold(x) for x in out}
    added = 0
    for raw in extra if isinstance(extra, list) else []:
        if not isinstance(raw, str):
            continue
        name = canonical_name(raw, entries)
        outside_gazetteer = name is None
        if outside_gazetteer:
            if added >= MAX_LLM_EXTRA:
                continue
            name = _clean_name(raw)
        key = fold(name) if name else ""
        if key and key not in seen:
            seen.add(key)
            out.append(name)
            added += outside_gazetteer
    return out


def merge_llm(tf: TextFeatures, data: dict[str, Any], persons: Sequence[GazEntry], brands: Sequence[GazEntry]) -> None:
    """Łączy wynik enrich() z heurystykami (w miejscu)."""
    tf.persons = _merge_names(tf.persons, data.get("persons"), persons)
    tf.brands = _merge_names(tf.brands, data.get("brands"), brands)
    allowed = narrative_labels()
    extra = [t for t in data.get("narratives") or [] if t in allowed and t not in tf.narratives]
    tf.narratives = (tf.narratives + extra)[: MAX_NARRATIVES + 1]
    for field in ("product", "promise", "cta"):
        value = data.get(field)
        if not getattr(tf, field) and isinstance(value, str) and value.strip():
            setattr(tf, field, _trim(value, 200))
    tf.llm_used = True


def _apply_llm(tf: TextFeatures, client: Any, persons: Sequence[GazEntry], brands: Sequence[GazEntry]) -> None:
    from slad.features.llm import enrich

    data = enrich(_llm_input(tf.input_text, tf.ocr_text, tf.page_text), client)
    if data:
        merge_llm(tf, data, persons, brands)


# ------------------------------------------------------------------ API


def text_features(material: Material, ocr_texts: Iterable[str] | None, page_text: str | None, ctx: Any) -> TextFeatures:
    """Cechy tekstowe materiału z tekstu wejściowego (Material.text), OCR obrazów i tekstu strony docelowej."""
    tcfg = getattr(getattr(ctx, "settings", None), "text", None) or TextSettings()
    input_text = _cap((material.text or "").strip(), MAX_STORED)
    ocr_text = _cap("\n".join(t.strip() for t in ocr_texts or () if isinstance(t, str) and t.strip()), MAX_STORED)
    page = _cap(page_text.strip(), MAX_STORED) if isinstance(page_text, str) else ""
    tf = TextFeatures(input_text=input_text, ocr_text=ocr_text, page_text=page)

    joined = "\n".join(p for p in (input_text[:MAX_INPUT_PART], ocr_text[:MAX_INPUT_PART], page) if p)
    if not joined.strip():
        return tf
    tf.text = normalize_text(joined)[:MAX_TEXT]
    folded = strip_diacritics(tf.text)
    display = _cap(_clean_lines(joined), MAX_TEXT + MAX_TEXT // 4)
    advertiser = " ".join((material.advertiser or "").split())
    gz_folded = f"{fold(advertiser)}\n{folded}" if advertiser else folded
    gz_cased = f"{advertiser}\n{display}" if advertiser else display
    ocr_folded = fold(ocr_text)[:5000] if ocr_text else ""
    persons = _safe("słownik osób", (), load_gazetteer, "persons")
    brands = _safe("słownik marek", (), load_gazetteer, "brands")

    tf.lang = _safe("język", None, detect_lang, tf.text)
    tf.keywords = _safe("słowa kluczowe", [], extract_keywords, tf.text, int(getattr(tcfg, "keywords", 12)))
    tf.persons = _safe("osoby", [], match_gazetteer, persons, gz_folded, gz_cased, ocr_folded)
    tf.brands = _safe("marki", [], match_gazetteer, brands, gz_folded, gz_cased, ocr_folded)
    tf.narratives = _safe("narracje", [], match_narratives, folded)
    tf.promise = _safe("obietnica", None, find_promise, display)
    tf.cta = _safe("wezwanie do działania", None, find_cta, display)
    tf.product = _safe("produkt", None, find_product, display, persons)
    tf.iocs = _safe("wskaźniki IOC", {}, extract_iocs, "\n".join(p for p in (input_text, ocr_text, page) if p))
    if getattr(ctx, "llm", None) is not None:
        _safe("LLM", None, _apply_llm, tf, ctx.llm, persons, brands)
    tf.minhash = _safe("MinHash", [], minhash_signature, tf.text, tcfg)
    return tf
