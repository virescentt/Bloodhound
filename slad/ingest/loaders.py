"""Rozpoznawanie formatu wejścia i mapowanie pól (PL/EN) → Material.

Obsługiwane wejścia (``load_dataset``):

- plik rekordów: ``.jsonl``/``.ndjson``, ``.json`` (lista, słownik z listą ``records``/``data``/``items``/…,
  słownik id → rekord), ``.csv``/``.tsv`` (separator i kodowanie wykrywane — także cp1250 z polskiego Excela),
  ``.xlsx``/``.xlsm`` (wszystkie arkusze, wiersz nagłówka wykrywany automatycznie, scalone komórki, hiperłącza,
  obrazy osadzone w arkuszu), ``.xls`` (jeśli jest xlrd);
- katalog z plikiem rekordów (wybierany ten z największą liczbą wierszy) — obrazy/HTML wskazane w rekordach są
  szukane względem katalogu pliku, w ``images/``, ``pages/`` itd. oraz po identyfikatorze (``<id>.png``);
- katalog bez pliku rekordów: każdy obraz, HTML/MHTML, TXT/EML to osobny materiał (pliki o tej samej nazwie
  bazowej — ``ad1.png`` + ``ad1.html`` + ``ad1.json`` — łączone), listy adresów (``urls.txt``, ``links.txt``,
  TXT z samymi URL-ami) dają po jednym materiale na adres, a podkatalog z kilkoma plikami różnych rodzajów
  (zrzut + HTML + opis) to JEDEN materiał;
- pojedynczy plik obrazu / HTML / MHTML / TXT / EML / .url.

Kolumny są rozpoznawane po synonimach (wielkość liter, polskie znaki i camelCase bez znaczenia, odmiana
polska — „Treść reklamy”, „URL obrazu”, „Nazwa kampanii”), a brakujące pola wykrywane po zawartości kolumn
(adresy URL, ścieżki obrazów, długie teksty). Zła wartość nigdy nie przerywa wczytywania: trafia do ``meta``,
a problem do ``meta["ingest_warnings"]``.
"""

from __future__ import annotations

import base64
import binascii
import csv
import email
import email.policy
import hashlib
import io
import json
import logging
import math
import os
import plistlib
import re
import sys
import tempfile
import unicodedata
import urllib.parse
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from slad.models import InfraFeatures, Material, SourceKind

log = logging.getLogger(__name__)

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"})
HTML_EXTS = frozenset({".html", ".htm", ".xhtml"})
MHTML_EXTS = frozenset({".mhtml", ".mht"})
TEXT_EXTS = frozenset({".txt", ".eml", ".log"})
LINK_EXTS = frozenset({".url", ".webloc"})
TABLE_EXTS = frozenset({".jsonl", ".ndjson", ".json", ".csv", ".tsv", ".xlsx", ".xlsm", ".xls"})
URL_LIST_NAMES = frozenset({"urls.txt", "links.txt", "url.txt", "link.txt", "linki.txt", "adresy.txt"})
IGNORED_STEMS = frozenset({"readme", "license", "licence", "changelog", "thumbs", "desktop", "metadata_schema"})
SKIP_DIRS = frozenset({"__macosx", "node_modules", "__pycache__", "css", "js", "fonts", ".git"})
EXTRACT_DIRNAME = ".slad_extracted"
MAX_TEXT_BYTES = 2_000_000
MAX_DEPTH = 10
EMPTY_VALUES = frozenset({"", "nan", "none", "null", "nil", "n/a", "na", "-", "--", "—", "brak", "brak danych", "?"})

# ---------------------------------------------------------------- synonimy kolumn
# Kolejność w krotce = priorytet (np. adres strony docelowej przed adresem reklamy).
SYNONYMS: dict[str, tuple[str, ...]] = {
    "id": (
        "id",
        "record_id",
        "ad_id",
        "uuid",
        "nr",
        "lp",
        "identyfikator",
        "material_id",
        "id_reklamy",
        "id_materialu",
        "numer",
        "no",
        "guid",
        "ad_archive_id",
    ),
    "url": (
        "landing_url",
        "landing_page",
        "landing",
        "final_url",
        "destination",
        "destination_url",
        "target_url",
        "url",
        "adres_url",
        "link",
        "strona_docelowa",
        "adres",
        "strona",
        "link_url",
        "website",
        "web",
        "www",
        "href",
        "ad_url",
    ),
    "text": (
        "text",
        "tekst",
        "tresc",
        "content",
        "ad_text",
        "body",
        "opis",
        "description",
        "message",
        "caption",
        "title",
        "tytul",
        "headline",
        "tekst_reklamy",
        "tresc_reklamy",
        "ad_body",
        "bodies",
        "post_text",
        "komunikat",
        "wiadomosc",
        "naglowek",
        "transcript",
        "transkrypcja",
        "ocr",
        "ocr_text",
    ),
    "image": (
        "image",
        "images",
        "image_path",
        "image_paths",
        "image_url",
        "image_urls",
        "screenshot",
        "screenshots",
        "screenshot_path",
        "zrzut",
        "zrzut_ekranu",
        "obraz",
        "obrazek",
        "img",
        "creative",
        "kreacja",
        "grafika",
        "zdjecie",
        "photo",
        "picture",
        "foto",
        "thumbnail",
        "miniatura",
        "file",
        "plik",
        "media",
    ),
    "html": (
        "html",
        "html_path",
        "page_html",
        "source_html",
        "html_file",
        "kod_html",
        "kod_strony",
        "zrodlo_strony",
        "page_source",
        "html_source",
        "saved_page",
        "zapis_strony",
        "mhtml",
    ),
    "platform": (
        "platform",
        "platforma",
        "source",
        "zrodlo",
        "network",
        "siec",
        "kanal",
        "channel",
        "serwis",
        "publisher_platforms",
    ),
    "advertiser": (
        "advertiser",
        "reklamodawca",
        "page_name",
        "page",
        "profil",
        "profile",
        "fanpage",
        "account",
        "konto",
        "author",
        "autor",
        "nadawca",
        "sender",
        "publisher",
        "nazwa_profilu",
        "advertiser_name",
        "username",
        "uzytkownik",
        "user",
    ),
    "date": (
        "date",
        "data",
        "observed_at",
        "timestamp",
        "created",
        "created_at",
        "first_seen",
        "published",
        "published_at",
        "data_publikacji",
        "data_obserwacji",
        "data_zgloszenia",
        "data_dodania",
        "datetime",
        "ad_creation_time",
        "ad_delivery_start_time",
        "creation",
        "reported_at",
        "zgloszono",
        "dzien",
        "day",
        "time",
        "czas",
        "seen",
        "last_seen",
    ),
    "label": (
        "label",
        "campaign",
        "kampania",
        "campaign_id",
        "cluster",
        "cluster_id",
        "group",
        "grupa",
        "campaign_name",
        "nazwa_kampanii",
        "id_kampanii",
        "klaster",
        "etykieta",
        "gold",
        "gold_label",
        "true_label",
    ),
    "infra": ("infra", "infrastructure", "infrastruktura", "infra_json"),
    "domain": ("domain", "domena", "hostname", "host", "landing_domain", "domena_docelowa"),
}

# Kolumny z gotowymi danymi infrastruktury → meta["infra"][klucz InfraFeatures] (tylko dokładne dopasowanie).
INFRA_COLUMNS: dict[str, str] = {
    "ip": "ips",
    "ips": "ips",
    "adres_ip": "ips",
    "ip_address": "ips",
    "adresy_ip": "ips",
    "asn": "asns",
    "asns": "asns",
    "as": "asns",
    "nameserver": "nameservers",
    "nameservers": "nameservers",
    "ns": "nameservers",
    "serwery_nazw": "nameservers",
    "registrar": "registrar",
    "rejestrator": "registrar",
    "domain_created": "domain_created",
    "creation_date": "domain_created",
    "data_rejestracji": "domain_created",
    "data_rejestracji_domeny": "domain_created",
    "domain_creation_date": "domain_created",
    "favicon_hash": "favicon_mmh3",
    "favicon_mmh3": "favicon_mmh3",
    "tls_fingerprint": "tls_fingerprint",
    "cert_sha256": "tls_fingerprint",
    "tls_issuer": "tls_issuer",
    "redirect_chain": "redirect_chain",
    "redirects": "redirect_chain",
    "przekierowania": "redirect_chain",
    "tracker_ids": "tracker_ids",
    "trackers": "tracker_ids",
    "mx": "mx",
    "http_status": "http_status",
    "status_code": "http_status",
}

WEAK_SYNONYMS = frozenset(
    {
        "page",
        "source",
        "zrodlo",
        "file",
        "plik",
        "media",
        "group",
        "grupa",
        "account",
        "konto",
        "profile",
        "profil",
        "user",
        "time",
        "czas",
        "day",
        "dzien",
        "web",
        "www",
        "strona",
        "adres",
        "network",
        "siec",
        "channel",
        "kanal",
        "seen",
        "host",
        "creative",
        "serwis",
    }
)
STOP_TOKENS = frozenset(
    {
        "email",
        "mail",
        "telefon",
        "phone",
        "tel",
        "ip",
        "asn",
        "whois",
        "nameserver",
        "ns",
        "dns",
        "count",
        "liczba",
        "score",
        "wynik",
        "status",
        "type",
        "typ",
        "kategoria",
        "category",
        "rodzaj",
        "lang",
        "jezyk",
        "language",
        "size",
        "rozmiar",
        "width",
        "height",
        "hash",
        "sha256",
        "md5",
        "phash",
        "uwagi",
        "notes",
        "comment",
        "komentarz",
    }
)
ID_TOKENS = frozenset({"id", "uuid", "nr", "lp", "numer", "no", "guid", "identyfikator"})
ID_CONTEXT = frozenset(
    {
        "ad",
        "record",
        "row",
        "material",
        "materialu",
        "item",
        "entry",
        "reklamy",
        "reklama",
        "rekordu",
        "rekord",
        "wpis",
        "wiersz",
        "sample",
        "probki",
        "nask",
        "unique",
        "unikalny",
        "post",
        "zgloszenia",
        "zgloszenie",
        "case",
        "sprawy",
        "archive",
        "ogloszenia",
    }
)
CONTENT_FIELDS = frozenset({"image", "html", "url", "text", "domain"})
PRECEDENCE = ("image", "html", "url", "date", "text", "label", "platform", "advertiser", "domain", "infra", "id")
META_KEYS = frozenset({"meta", "metadata", "metadane"})
# pola meta zapisywane przez sam loader — przy ponownym wczytaniu (np. materials.jsonl) kopiowane wprost
PASSTHROUGH_META = frozenset(
    {
        "source_file",
        "source_row",
        "sheet",
        "source_path",
        "ingest_warnings",
        "missing_images",
        "url_raw",
        "extra_html",
    }
)
PLATFORM_ALIASES = {
    "fb": "facebook",
    "facebook.com": "facebook",
    "meta": "facebook",
    "ig": "instagram",
    "insta": "instagram",
    "instagram.com": "instagram",
    "yt": "youtube",
    "youtube.com": "youtube",
    "tt": "tiktok",
    "tik tok": "tiktok",
    "tiktok.com": "tiktok",
    "google ads": "google",
    "adwords": "google",
    "gads": "google",
    "twitter": "x",
    "x.com": "x",
    "messenger": "facebook",
}
_SUFFIXES = frozenset({"s", "es", "y", "u", "i", "e", "a", "ow", "om", "ach", "ami", "ie", "ii", "ek", "ki"})
_SOURCE_KIND_VALUES = frozenset(k.value for k in SourceKind)


def norm_key(name: Any) -> str:
    """Nazwa kolumny → postać porównywalna: małe litery, bez polskich znaków, ``_`` jako separator."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))  # camelCase → camel_Case
    s = s.replace("ł", "l").replace("Ł", "L")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


_EXACT: dict[str, tuple[str, int]] = {}
for _f, _syns in SYNONYMS.items():
    for _rank, _s in enumerate(_syns):
        _EXACT.setdefault(norm_key(_s), (_f, _rank))
for _k, _v in INFRA_COLUMNS.items():
    _EXACT.setdefault(_k, ("infra:" + _v, 50))
_SINGLE = [(s, f) for s, (f, _) in _EXACT.items() if "_" not in s and not f.startswith("infra:")]
_MULTI = [(s, f) for s, (f, _) in _EXACT.items() if "_" in s and not f.startswith("infra:")]


def _inflect_match(token: str, syn: str) -> bool:
    """Dopasowanie z uwzględnieniem prostej odmiany (obrazu, kampanii, reklamodawcy, linki, daty)."""
    if token == syn:
        return True
    if len(syn) >= 3 and token.startswith(syn) and token[len(syn) :] in _SUFFIXES:
        return True
    return len(syn) >= 4 and syn[-1] in "aeiouy" and token.startswith(syn[:-1]) and token[len(syn) - 1 :] in _SUFFIXES


def map_column(name: Any) -> tuple[str | None, int]:
    """Nazwa kolumny → (pole, priorytet). Pola: id, url, text, image, html, platform, advertiser, date, label,
    infra, domain oraz ``infra:<klucz>``; ``None`` = kolumna trafia do meta. Mniejszy priorytet = lepszy."""
    k = norm_key(name)
    if not k:
        return None, 999
    if k in _EXACT:
        return _EXACT[k]
    tokens = k.split("_")
    if any(t in STOP_TOKENS for t in tokens):
        return None, 999
    strong: list[tuple[int, str]] = []  # (pozycja tokenu, pole)
    weak: list[tuple[int, str]] = []
    padded = f"_{k}_"
    for syn, f in _MULTI:
        pos = padded.find(f"_{syn}_")
        if pos >= 0:
            (weak if syn in WEAK_SYNONYMS else strong).append((padded[:pos].count("_"), f))
    has_id = any(t in ID_TOKENS for t in tokens)
    for i, t in enumerate(tokens):
        if t in ID_TOKENS:
            continue
        for syn, f in _SINGLE:
            if f != "id" and _inflect_match(t, syn):
                (weak if syn in WEAK_SYNONYMS else strong).append((i, f))
    cands = strong or weak
    if has_id:  # „creative_id”, „page_id” — identyfikator, nie treść
        cands = [(i, f) for i, f in cands if f not in CONTENT_FIELDS]
    if cands:
        fields = {f for _, f in cands}
        if {"url", "text"} <= fields and not fields & {"image", "html"}:
            # „ad_creative_link_titles” (tytuły linku) → tekst; „text_url” → url: decyduje ostatni z nich
            best = max((c for c in cands if c[1] in ("url", "text")), key=lambda c: c[0])[1]
        else:  # adres obrazu/strony („url_obrazu”, „screenshot_url”) to odnośnik do obrazu/HTML
            best = min(fields, key=PRECEDENCE.index)
        return best, 100 + PRECEDENCE.index(best)
    if has_id and all(t in ID_TOKENS or t in ID_CONTEXT for t in tokens):
        return "id", 150
    return None, 999


# ---------------------------------------------------------------- wartości: URL, daty, teksty

_SCHEME_RE = re.compile(r"^([a-z][a-z0-9+.-]*)://", re.I)
_DOMAIN_RE = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{1,62}\.?(?::\d{1,5})?(?:[/?#]\S*)?$", re.I
)
_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}(?::\d{1,5})?(?:[/?#]\S*)?$")
_FILE_LIKE_TLDS = frozenset(
    e.lstrip(".") for e in IMAGE_EXTS | HTML_EXTS | MHTML_EXTS | TEXT_EXTS | TABLE_EXTS | LINK_EXTS
) | {"php", "asp", "aspx", "jsp", "js", "css", "pdf", "zip", "exe", "doc", "docx", "mov", "mp4", "py", "md"}
_MARKUP_RE = re.compile(
    r"</?(?:p|div|span|br|a|b|i|u|strong|em|h[1-6]|ul|ol|li|img|html|body|table|tr|td)\b[^>]*>", re.I
)


def refang(value: str) -> str:
    """Odwraca „rozbrojone” adresy z raportów CTI: hxxps://scam[.]pl → https://scam.pl."""
    s = re.sub(r"^h[x]{2}p(s?)", r"http\1", value.strip(), flags=re.I)
    for a, b in (
        ("[.]", "."),
        ("(.)", "."),
        ("{.}", "."),
        ("[dot]", "."),
        ("(dot)", "."),
        ("[:]", ":"),
        ("[://]", "://"),
    ):
        s = s.replace(a, b)
    return s


def normalize_url(value: Any) -> str | None:
    """Wartość → adres http(s) albo ``None``. Dodaje schemat do samej domeny (``zysk.pl/x`` → ``http://zysk.pl/x``)."""
    if value is None:
        return None
    s = refang(str(value)).strip().strip("<>\"'")
    if not s or any(c.isspace() for c in s):
        return None
    if s.startswith("//"):
        s = "https:" + s
    m = _SCHEME_RE.match(s)
    if m:
        scheme = m.group(1).lower()
        rest = s[m.end() :]
        host = re.split(r"[/?#]", rest, maxsplit=1)[0]
        if scheme not in ("http", "https") or not host or ("." not in host and ":" not in host and host != "localhost"):
            return None
        return f"{scheme}://{rest}"
    if s.lower().startswith("www.") or _DOMAIN_RE.match(s) or _IPV4_RE.match(s):
        host = re.split(r"[/?#:]", s, maxsplit=1)[0].rstrip(".")
        tld = host.rsplit(".", 1)[-1].lower()
        if not s.lower().startswith("www.") and (tld in _FILE_LIKE_TLDS or (tld.isdigit() and not _IPV4_RE.match(s))):
            return None
        return "http://" + s
    return None


def _looks_like_url_strict(v: str) -> bool:
    s = refang(v).strip().lower()
    return bool(re.match(r"^(https?://|www\.)\S+$", s))


def _ref_ext(v: str) -> str:
    path = urllib.parse.urlsplit(v.strip()).path if "://" in v else v.strip()
    return Path(path.replace("\\", "/")).suffix.lower()


def _looks_like_image_ref(v: str) -> bool:
    return v.strip().lower().startswith("data:image/") or _ref_ext(v) in IMAGE_EXTS


def _looks_like_html_ref(v: str) -> bool:
    return _ref_ext(v) in HTML_EXTS | MHTML_EXTS


_PL_MONTHS = {
    "styczen": 1,
    "stycznia": 1,
    "luty": 2,
    "lutego": 2,
    "marzec": 3,
    "marca": 3,
    "kwiecien": 4,
    "kwietnia": 4,
    "maj": 5,
    "maja": 5,
    "czerwiec": 6,
    "czerwca": 6,
    "lipiec": 7,
    "lipca": 7,
    "sierpien": 8,
    "sierpnia": 8,
    "wrzesien": 9,
    "wrzesnia": 9,
    "pazdziernik": 10,
    "pazdziernika": 10,
    "listopad": 11,
    "listopada": 11,
    "grudzien": 12,
    "grudnia": 12,
    "sty": 1,
    "lut": 2,
    "mar": 3,
    "kwi": 4,
    "cze": 6,
    "lip": 7,
    "sie": 8,
    "wrz": 9,
    "paz": 10,
    "lis": 11,
    "gru": 12,
}
_DATE_FORMATS = (
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%d.%m.%Y",
    "%d.%m.%y",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%d-%m-%Y",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y%m%d",
)
_EXCEL_EPOCH = datetime(1899, 12, 30)


def _naive_utc(dt: datetime) -> datetime:
    return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo else dt


def _from_number(x: float) -> datetime | None:
    try:
        if x > 1e11:  # epoka w milisekundach
            return datetime.fromtimestamp(x / 1000, tz=UTC).replace(tzinfo=None)
        if x > 1e8:  # epoka w sekundach
            return datetime.fromtimestamp(x, tz=UTC).replace(tzinfo=None)
        if 20000 < x < 80000:  # numer seryjny daty Excela
            return _EXCEL_EPOCH + timedelta(days=x)
    except (OverflowError, OSError, ValueError):
        return None
    return None


def parse_date(value: Any) -> datetime | None:
    """Łagodne parsowanie dat: ISO, dd.mm.rrrr [gg:mm], polskie nazwy miesięcy, epoka (s/ms), numer Excela.

    Wynik bez strefy czasowej (UTC), żeby daty z różnych źródeł dało się porównywać.
    """
    if value is None or isinstance(value, bool):
        return None
    dt: datetime | None = None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    elif isinstance(value, int | float):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        dt = _from_number(float(value))
    else:
        s = " ".join(str(value).strip().split())
        if not s or s.lower() in EMPTY_VALUES:
            return None
        if re.fullmatch(r"\d{9,13}(\.\d+)?|\d{5}(\.\d+)?", s):
            dt = _from_number(float(s))
        if dt is None:
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00") if s.endswith("Z") else s)
            except ValueError:
                dt = None
        if dt is None:
            for fmt in _DATE_FORMATS:
                try:
                    dt = datetime.strptime(s, fmt)
                    break
                except ValueError:
                    continue
        if dt is None:
            dt = _parse_polish_date(s)
        # dateutil jako ostatnia deska ratunku — tylko gdy napis ma kształt daty (inaczej „3” → dziś z dniem 3)
        if dt is None and (
            len(re.findall(r"\d+", s)) >= 2 or (re.search(r"[^\W\d_]{3,}", s) and re.search(r"\d{4}", s))
        ):
            try:
                from dateutil import parser as du_parser

                dt = du_parser.parse(s, dayfirst=True)
            except (ImportError, ValueError, OverflowError, TypeError):
                dt = None
    if dt is None:
        return None
    dt = _naive_utc(dt)
    return dt if 1990 <= dt.year <= 2100 else None


def _parse_polish_date(s: str) -> datetime | None:
    m = re.search(r"(\d{1,2})\s+([^\W\d_]+)\.?\s+(\d{4})(?:\D+(\d{1,2}):(\d{2}))?", s)
    if not m:
        return None
    month = _PL_MONTHS.get(norm_key(m.group(2)))
    if not month:
        return None
    try:
        return datetime(int(m.group(3)), month, int(m.group(1)), int(m.group(4) or 0), int(m.group(5) or 0))
    except ValueError:
        return None


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str):
        return v.strip().lower() in EMPTY_VALUES
    if isinstance(v, list | tuple | set | dict):
        return len(v) == 0
    return False


def _scalar_str(v: Any) -> str | None:
    """Wartość skalarna → napis (1.0 → „1”); puste → None."""
    if _is_empty(v):
        return None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, datetime | date):
        return v.isoformat()
    s = str(v).strip()
    if re.fullmatch(r"-?\d+\.0", s):
        s = s[:-2]
    return s or None


def _json_safe(v: Any) -> Any:
    """Meta musi dać się zapisać jako JSON (typy numpy, daty, NaN, bajty)."""
    if v is None or isinstance(v, bool | int | str):
        return v
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(v, datetime | date):
        return v.isoformat()
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, list | tuple | set):
        return [_json_safe(x) for x in v]
    if isinstance(v, bytes):
        return f"<{len(v)} B>"
    item = getattr(v, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(v)


def _strip_markup(text: str) -> str:
    if not _MARKUP_RE.search(text):
        return text
    try:
        from bs4 import BeautifulSoup

        return " ".join(BeautifulSoup(text, "lxml").get_text(" ").split())
    except Exception:  # noqa: BLE001 — w najgorszym razie zostawiamy surowy tekst
        return re.sub(r"<[^>]+>", " ", text)


def _as_text(v: Any) -> str | None:
    if _is_empty(v):
        return None
    if isinstance(v, list | tuple):
        parts = [_as_text(x) for x in v]
        s = "\n".join(p for p in parts if p)
    elif isinstance(v, dict):
        s = json.dumps(_json_safe(v), ensure_ascii=False)
    else:
        s = _scalar_str(v) or ""
    s = _strip_markup(s).strip()
    return s or None


def _as_list(v: Any, split: bool = True) -> list[str]:
    """Lista wartości z komórki: lista JSON, napis rozdzielany ; | , lub nowymi wierszami."""
    if _is_empty(v):
        return []
    if isinstance(v, list | tuple | set):
        out: list[str] = []
        for x in v:
            if isinstance(x, dict):  # np. {"path": "a.png", "caption": "…"}
                for key in ("path", "url", "src", "file", "href", "name"):
                    if isinstance(x.get(key), str):
                        out.append(x[key])
                        break
            else:
                out.extend(_as_list(x, split))
        return out
    if isinstance(v, dict):
        return _as_list(list(v.values()), split)
    s = str(v).strip()
    if s.lower().startswith("data:") or not split:
        return [s]
    if s.startswith("[") and s.endswith("]"):  # lista JSON zapisana w komórce CSV
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return _as_list(parsed, split)
        except json.JSONDecodeError:
            pass
    parts = [p.strip() for p in re.split(r"[;|\n]+", s) if p.strip()]
    if len(parts) == 1 and "," in s:
        pieces = [p.strip() for p in s.split(",") if p.strip()]
        if len(pieces) > 1 and all(_ref_ext(p) or p.lower().startswith(("http", "www.")) for p in pieces):
            parts = pieces
    return parts


# ---------------------------------------------------------------- pliki pomocnicze


def _read_text(path: Path, limit: int = MAX_TEXT_BYTES) -> str:
    data = path.read_bytes()[:limit]
    for enc in ("utf-8-sig", "cp1250"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _safe_name(s: str, n: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:n].strip("._") or "plik"


class _Extractor:
    """Zapis plików pochodnych (obrazy osadzone w XLSX, HTML z MHTML / z komórki, data: URI).

    Domyślnie ``<katalog źródła>/.slad_extracted/<nazwa>/``, gdy brak praw zapisu — katalog tymczasowy.
    Nazwy zależą od treści, więc ponowne wczytanie daje te same ścieżki.
    """

    def __init__(self, anchor: Path):
        self.anchor = anchor
        self._dir: Path | None = None

    def _fallback(self) -> Path:
        h = hashlib.sha1(str(self.anchor).encode()).hexdigest()[:12]
        d = Path(tempfile.gettempdir()) / "slad_extracted" / h
        d.mkdir(parents=True, exist_ok=True)
        return d

    def dir(self) -> Path:
        if self._dir is None:
            base = self.anchor.parent if self.anchor.is_file() else self.anchor
            d = base / EXTRACT_DIRNAME / (_safe_name(self.anchor.stem) if self.anchor.is_file() else "")
            try:
                d.mkdir(parents=True, exist_ok=True)
                self._dir = d
            except OSError:
                self._dir = self._fallback()
        return self._dir

    def save(self, data: bytes, ext: str, hint: str) -> Path:
        name = f"{_safe_name(hint)}-{hashlib.sha1(data).hexdigest()[:12]}{ext}"
        for attempt in range(2):
            p = (self.dir() if attempt == 0 else self._fallback()) / name
            try:
                if not p.exists():
                    p.write_bytes(data)
                return p.resolve()
            except OSError:
                self._dir = self._fallback()
        raise OSError(f"nie można zapisać pliku pomocniczego {name}")


class _Resolver:
    """Odnajduje pliki wskazane w rekordach (ścieżki względne, Windowsowe, same nazwy, ``<id>.png``)."""

    SUBDIRS = (
        "images",
        "image",
        "img",
        "imgs",
        "screenshots",
        "screens",
        "zrzuty",
        "obrazy",
        "media",
        "files",
        "pliki",
        "pages",
        "html",
        "strony",
        "assets",
        "data",
    )

    def __init__(self, base: Path):
        self.base = base
        self._names: dict[str, list[Path]] | None = None
        self._stems: dict[str, list[Path]] = {}
        self._prefixes: dict[str, list[Path]] = {}

    def _ensure(self) -> None:
        if self._names is not None:
            return
        names: dict[str, list[Path]] = defaultdict(list)
        stems: dict[str, list[Path]] = defaultdict(list)
        prefixes: dict[str, list[Path]] = defaultdict(list)
        count = 0
        for dirpath, dirnames, filenames in os.walk(self.base):
            depth = len(Path(dirpath).relative_to(self.base).parts)
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d.lower() not in SKIP_DIRS)
            if depth >= 5:
                dirnames[:] = []
            for fn in sorted(filenames):
                if fn.startswith("."):
                    continue
                p = Path(dirpath) / fn
                stem = p.stem.lower()
                names[fn.lower()].append(p)
                stems[stem].append(p)
                for m in re.finditer(r"[_\-. ]", stem):
                    prefixes[stem[: m.start()]].append(p)
                count += 1
            if count > 300_000:
                log.warning("Bardzo duży katalog %s — indeks plików ograniczony", self.base)
                break
        self._names, self._stems, self._prefixes = names, stems, prefixes

    def find(self, ref: str) -> Path | None:
        ref = ref.strip().strip("\"'")
        if not ref:
            return None
        if ref.lower().startswith("file://"):
            ref = urllib.parse.unquote(urllib.parse.urlsplit(ref).path)
        variants = [ref]
        if "%" in ref:
            variants.append(urllib.parse.unquote(ref))
        for r in variants:
            p = Path(r).expanduser()
            cands = [p] if p.is_absolute() else [self.base / p, *(self.base / sub / p.name for sub in self.SUBDIRS)]
            for c in cands:
                try:
                    if c.is_file():
                        return c.resolve()
                except OSError:
                    continue
        self._ensure()
        assert self._names is not None
        name = re.split(r"[\\/]", variants[-1])[-1].lower()
        hits = self._names.get(name)
        if hits:
            return hits[0].resolve()
        if "." not in name:
            for h in self._stems.get(name, []):
                if h.suffix.lower() in IMAGE_EXTS | HTML_EXTS:
                    return h.resolve()
        return None

    def by_id(self, mid: str, exts: frozenset[str]) -> list[Path]:
        """Pliki nazwane identyfikatorem materiału: ``<id>.png``, ``<id>_1.png``, ``<id>-2.jpg``."""
        self._ensure()
        key = mid.strip().lower()
        if not key:
            return []
        hits = [p for p in self._stems.get(key, []) if p.suffix.lower() in exts]
        hits += [p for p in self._prefixes.get(key, []) if p.suffix.lower() in exts and p not in hits]
        return sorted({p.resolve() for p in hits})


# ---------------------------------------------------------------- szkic materiału


@dataclass
class _Draft:
    id: str | None = None
    urls: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    images: list[Path] = field(default_factory=list)
    html: Path | None = None
    platform: str | None = None
    advertiser: str | None = None
    observed_at: datetime | None = None
    label: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    image_urls: list[str] = field(default_factory=list)
    infra: dict[str, Any] = field(default_factory=dict)
    infra_cols: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    rel_dir: str | None = None  # katalog źródłowy względem korzenia zbioru (tryb katalogu)
    fingerprint: str = ""  # zapasowy materiał do skrótu id

    def add_url(self, u: str) -> None:
        if u not in self.urls:
            self.urls.append(u)

    def add_text(self, t: str) -> None:
        if t and t not in self.texts:
            self.texts.append(t)

    def add_image(self, p: Path) -> None:
        if p not in self.images:
            self.images.append(p)


@dataclass
class _Ctx:
    root: Path
    extractor: _Extractor
    resolvers: dict[Path, _Resolver] = field(default_factory=dict)

    def resolver(self, base: Path) -> _Resolver:
        if base not in self.resolvers:
            self.resolvers[base] = _Resolver(base)
        return self.resolvers[base]


_INFRA_LIST_FIELDS = frozenset(
    name for name, f in InfraFeatures.model_fields.items() if "list" in str(f.annotation).lower()
)
_INFRA_ALIASES = {
    "ip": "ips",
    "asn": "asns",
    "ns": "nameservers",
    "favicon_hash": "favicon_mmh3",
    "redirects": "redirect_chain",
    "created": "domain_created",
    "creation_date": "domain_created",
    "trackers": "tracker_ids",
}


def _coerce_infra(d: dict[str, Any]) -> dict[str, Any]:
    """Lekka normalizacja gotowych danych infrastruktury: napis zamiast listy, ASN jako liczba, hash jako tekst."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = _INFRA_ALIASES.get(norm_key(k), str(k))
        if _is_empty(v):
            continue
        if key in _INFRA_LIST_FIELDS:
            if isinstance(v, str):
                # ASN ma spację w nazwie („AS13335 CLOUDFLARENET”) — dzielimy tylko po ; | i nowych wierszach
                sep = r"[;|\n]+" if key == "asns" else r"\s*(?:->|→|=>|[,;|\s])\s*"
                v = [x.strip() for x in re.split(sep, v) if x.strip()]
            elif not isinstance(v, list):
                v = [v]
            if key == "asns":
                v = [f"AS{x}" if str(x).isdigit() else str(x) for x in v]
        elif key in ("favicon_mmh3", "http_status") and not isinstance(v, int):
            try:
                v = int(float(v))
            except (TypeError, ValueError):
                continue
        out[key] = v
    return out


def _parse_infra(v: Any) -> dict[str, Any] | None:
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip().startswith("{"):
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _norm_platform(v: Any) -> str | None:
    if isinstance(v, list | tuple):
        vals = [_norm_platform(x) for x in v]
        vals = [x for x in vals if x]
        return ", ".join(dict.fromkeys(vals)) or None
    s = _scalar_str(v)
    if not s:
        return None
    s = s.strip().lower()
    if s in _SOURCE_KIND_VALUES:  # pole „source” z Material — to nie platforma
        return None
    return PLATFORM_ALIASES.get(s, s)


# ---------------------------------------------------------------- rekord → szkic


@dataclass
class _Plan:
    columns: dict[str, tuple[str, int]]  # kolumna → (pole, priorytet)
    order: dict[str, int]


def _plan_columns(columns: list[str], rows: list[dict[str, Any]]) -> _Plan:
    """Mapowanie kolumn tabeli: synonimy, a dla brakujących pól — rozpoznanie po zawartości."""
    plan: dict[str, tuple[str, int]] = {}
    for col in columns:
        f, pr = map_column(col)
        if f:
            plan[col] = (f, pr)
    present = {f for f, _ in plan.values()}
    unmapped = [c for c in columns if c not in plan and norm_key(c) not in META_KEYS]
    sample = rows[:300]

    def values(col: str) -> list[str]:
        out = []
        for r in sample:
            v = r.get(col)
            for x in v if isinstance(v, list) else [v]:  # listy (JSON) — każdy element osobno
                if isinstance(x, str) and not _is_empty(x):
                    out.append(x)
        return out

    for fld, test in (
        ("image", _looks_like_image_ref),
        ("html", _looks_like_html_ref),
        ("url", _looks_like_url_strict),
    ):
        if fld in present or (fld == "url" and "domain" in present):
            continue
        for col in list(unmapped):
            vals = values(col)
            if vals and sum(1 for v in vals if test(v)) / len(vals) >= 0.8:
                plan[col] = (fld, 200)
                present.add(fld)
                unmapped.remove(col)
                break
    if "text" not in present:
        best: tuple[float, str] | None = None
        for col in unmapped:
            vals = values(col)
            if not vals:
                continue
            avg = sum(len(v) for v in vals) / len(vals)
            spaced = sum(1 for v in vals if " " in v.strip()) / len(vals)
            if avg >= 40 and spaced >= 0.6 and (best is None or avg > best[0]):
                best = (avg, col)
        if best:
            plan[best[1]] = ("text", 200)
    return _Plan(plan, {c: i for i, c in enumerate(columns)})


def _flatten(row: dict[str, Any], prefix: str = "", depth: int = 0) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        key = f"{prefix}{k}"
        nk = norm_key(k)
        if isinstance(v, dict) and depth < 2 and nk not in META_KEYS and map_column(k)[0] != "infra":
            out.update(_flatten(v, f"{key}_", depth + 1))
        else:
            out[key] = v
    return out


def _apply_row(d: _Draft, row: dict[str, Any], plan: _Plan, ctx: _Ctx, base: Path) -> None:
    """Przenosi pola rekordu do szkicu (wartości odrzucone przez walidację → meta)."""
    by_field: dict[str, list[tuple[int, int, str, Any]]] = defaultdict(list)
    for col, value in row.items():
        if _is_empty(value):
            continue
        nk = norm_key(col)
        if nk in META_KEYS and isinstance(value, dict):
            _apply_nested_meta(d, value, ctx, base)
            continue
        if col in plan.columns:
            f, pr = plan.columns[col]
            by_field[f].append((pr, plan.order.get(col, 999), col, value))
        else:
            d.meta[_meta_key(col, d)] = value
    for f in by_field:
        by_field[f].sort(key=lambda t: (t[0], t[1]))

    for _, _, _col, v in by_field.get("id", []):
        s = _scalar_str(v)
        if s:
            d.id = d.id or s
            break
    for _, _, col, v in by_field.get("url", []):
        added = False
        for item in _as_list(v):
            u = normalize_url(item)
            if u:
                d.add_url(u)
                added = True
        if not added:
            d.meta[_meta_key(col, d)] = v
    for _, _, col, v in by_field.get("domain", []):
        doms = [x for x in _as_list(v) if normalize_url(x)]
        if doms:
            d.domains.extend(doms)
        d.meta[_meta_key(col, d)] = v
    for _, _, _col, v in by_field.get("text", []):
        t = _as_text(v)
        if t:
            d.add_text(t)
    for _, _, col, v in by_field.get("image", []):
        for ref in _as_list(v):
            _add_image_ref(d, ref, ctx, base, col)
    for _, _, col, v in by_field.get("html", []):
        if d.html is None:
            _set_html(d, v, ctx, base, col)
        else:
            d.meta[_meta_key(col, d)] = v
    for _, _, col, v in by_field.get("platform", []):
        if isinstance(v, str) and v.strip().lower() in _SOURCE_KIND_VALUES:
            continue  # pole „source” z Material (record/image/…) — nie platforma
        p = _norm_platform(v)
        if p and d.platform is None:
            d.platform = p
        elif not p or d.platform != p:
            d.meta[_meta_key(col, d)] = v
    for _, _, col, v in by_field.get("advertiser", []):
        s = _as_text(v)
        if s and d.advertiser is None:
            d.advertiser = s[:300]
        else:
            d.meta[_meta_key(col, d)] = v
    for _, _, col, v in by_field.get("date", []):
        dt = parse_date(v)
        if dt and d.observed_at is None:
            d.observed_at = dt
        elif dt is None or d.observed_at != dt:
            d.meta[_meta_key(col, d)] = v
    for _, _, col, v in by_field.get("label", []):
        s = _scalar_str(v)
        if s and d.label is None:
            d.label = s
        elif s and s != d.label:  # druga kolumna etykiety — nie do meta (wyciek etykiety do cech)
            log.debug("Pominięto dodatkową kolumnę etykiety %s", col)
    for _, _, col, v in by_field.get("infra", []):
        parsed = _parse_infra(v)
        if parsed is not None:
            d.infra.update(parsed)
        else:
            d.meta[_meta_key(col, d)] = v
    for f, items in by_field.items():
        if f.startswith("infra:"):
            key = f.split(":", 1)[1]
            for _, _, _col, v in items:
                d.infra_cols.setdefault(key, v)


def _apply_nested_meta(d: _Draft, meta: dict[str, Any], ctx: _Ctx, base: Path) -> None:
    """Zagnieżdżone ``meta`` (np. ponowne wczytanie materials.jsonl): infra, image_urls, urls + reszta wprost."""
    for k, v in meta.items():
        if _is_empty(v):
            continue
        nk = norm_key(k)
        if nk == "infra" and isinstance(_parse_infra(v), dict):
            d.infra.update(_parse_infra(v) or {})
        elif nk == "image_urls":
            d.image_urls.extend(x for x in _as_list(v) if x not in d.image_urls)
        elif nk == "urls":
            for x in _as_list(v):
                u = normalize_url(x)
                if u:
                    d.add_url(u)
        elif nk in PASSTHROUGH_META:
            d.meta.setdefault(k, v)
        else:
            d.meta.setdefault(_meta_key(k, d), v)


_RESERVED_META = frozenset({"infra", "image_urls", "urls"}) | PASSTHROUGH_META


def _meta_key(col: str, d: _Draft) -> str:
    key = str(col)
    return f"raw_{key}" if key in _RESERVED_META else key


def _add_image_ref(d: _Draft, ref: str, ctx: _Ctx, base: Path, col: str) -> None:
    ref = ref.strip()
    if not ref:
        return
    low = ref.lower()
    if low.startswith("data:"):
        m = re.match(r"data:([\w/+.-]+)?(;base64)?,(.*)$", ref, re.S)
        if m and m.group(2):
            try:
                data = base64.b64decode(m.group(3), validate=False)
                ext = "." + (m.group(1) or "image/png").split("/")[-1].replace("jpeg", "jpg").split("+")[0]
                d.add_image(ctx.extractor.save(data, ext if ext in IMAGE_EXTS else ".png", d.id or "obraz"))
                return
            except (binascii.Error, ValueError, OSError) as e:
                d.warnings.append(f"nieczytelny obraz data: URI ({e})")
                return
        d.warnings.append("nieobsługiwany obraz data: URI")
        return
    if low.startswith(("http://", "https://", "//")):
        if ref not in d.image_urls:
            d.image_urls.append(ref)
        return
    if _ref_ext(ref) in HTML_EXTS | MHTML_EXTS:  # kolumna „plik” wskazuje HTML
        if d.html is None:
            _set_html(d, ref, ctx, base, col)
        return
    p = ctx.resolver(base).find(ref)
    if p is None:
        d.meta.setdefault("missing_images", []).append(ref)
        d.warnings.append(f"brak pliku obrazu: {ref}")
        return
    if p.suffix.lower() not in IMAGE_EXTS:
        d.warnings.append(f"plik nie jest obrazem: {p.name}")
        return
    d.add_image(p)


def _set_html(d: _Draft, v: Any, ctx: _Ctx, base: Path, col: str) -> None:
    s = v if isinstance(v, str) else _scalar_str(v)
    if not s:
        return
    head = s.lstrip()[:300].lower()
    if head.startswith(("<!doctype", "<html")) or ("<" in head and "</" in s and "\n" in s.strip()):
        try:
            d.html = ctx.extractor.save(s.encode("utf-8"), ".html", d.id or "strona")
        except OSError as e:
            d.warnings.append(f"nie zapisano HTML z rekordu: {e}")
        return
    if s.lower().startswith(("http://", "https://")):  # adres zamiast pliku
        u = normalize_url(s)
        if u:
            d.add_url(u)
        return
    p = ctx.resolver(base).find(s)
    if p is None:
        d.warnings.append(f"brak pliku HTML: {s}")
        d.meta[_meta_key(col, d)] = s
        return
    if p.suffix.lower() in MHTML_EXTS:
        _apply_mhtml(d, p, ctx)
    else:
        d.html = p


def _records_to_drafts(
    rows: list[dict[str, Any]],
    columns: list[str] | None,
    base: Path,
    ctx: _Ctx,
    source: Path,
    *,
    sheet: str | None = None,
    row_numbers: list[int] | None = None,
    embedded: dict[int, list[Path]] | None = None,
) -> list[_Draft]:
    flat = [_flatten(r) for r in rows]
    if columns is None:
        seen: dict[str, None] = {}
        for r in flat:
            for k in r:
                seen.setdefault(k, None)
        columns = list(seen)
    plan = _plan_columns(columns, flat)
    if not any(f in {"url", "text", "image", "html", "domain"} for f, _ in plan.columns.values()):
        log.warning("%s: nie rozpoznano kolumn z treścią (kolumny: %s)", source.name, ", ".join(columns[:20]))
    drafts: list[_Draft] = []
    for i, row in enumerate(flat):
        d = _Draft()
        try:
            _apply_row(d, row, plan, ctx, base)
        except Exception as e:  # noqa: BLE001 — jeden zły rekord nie przerywa wczytywania
            log.warning("%s: rekord %d pominięty częściowo: %s", source.name, i + 1, e)
            d.warnings.append(f"błąd wczytywania rekordu: {e}")
        rown = row_numbers[i] if row_numbers else i + 1
        if embedded and rown in embedded:
            for p in embedded[rown]:
                d.add_image(p)
        d.meta.setdefault("source_file", source.name)
        d.meta.setdefault("source_row", rown)
        if sheet:
            d.meta.setdefault("sheet", sheet)
        d.fingerprint = json.dumps(_json_safe(row), sort_keys=True, ensure_ascii=False)[:5000]
        _match_by_id(d, ctx, base)
        drafts.append(d)
    return drafts


def _match_by_id(d: _Draft, ctx: _Ctx, base: Path) -> None:
    """Brak wskazanego obrazu / HTML → pliki nazwane identyfikatorem rekordu (``<id>.png``)."""
    if not d.id:
        return
    r = ctx.resolver(base)
    if not d.images:
        for p in r.by_id(d.id, IMAGE_EXTS):
            d.add_image(p)
    if d.html is None:
        hits = r.by_id(d.id, HTML_EXTS)
        if hits:
            d.html = hits[0]


# ---------------------------------------------------------------- czytniki tabel


def _sniff_delimiter(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            counts = {}
            for sep in (",", ";", "\t", "|"):
                stripped = re.sub(r'"[^"]*"', "", line)
                counts[sep] = stripped.count(sep)
            best = max(counts, key=lambda k: counts[k])
            return best if counts[best] > 0 else ","
    return ","


def _detect_header(rows: list[list[Any]]) -> int | None:
    """Indeks wiersza nagłówka (pierwsze 10 wierszy) albo None, gdy tabela nie ma nagłówka."""

    def header_like(c: Any) -> bool:  # nazwa kolumny: krótka, bez adresów, ścieżek i zdań
        if not isinstance(c, str) or len(c) > 40 or _looks_like_url_strict(c) or _ref_ext(c):
            return False
        return len(norm_key(c).split("_")) <= 5

    best_i, best_score = 0, 0
    for i, row in enumerate(rows[:10]):
        cells = [c for c in row if not _is_empty(c)]
        score = sum(1 for c in cells if header_like(c) and map_column(c)[0])
        if score > best_score:
            best_i, best_score = i, score
    if best_score:
        return best_i
    first = next((r for r in rows if any(not _is_empty(c) for c in r)), None)
    if first is None:
        return None
    looks_data = any(
        isinstance(c, str) and (_looks_like_url_strict(c) or _ref_ext(c) in IMAGE_EXTS | HTML_EXTS or len(c) > 60)
        for c in first
    )
    return None if looks_data else rows.index(first)


def _grid_to_rows(grid: list[list[Any]]) -> tuple[list[str], list[dict[str, Any]], list[int]]:
    """Siatka komórek → (kolumny, rekordy, numery wierszy 1-based)."""
    if not grid:
        return [], [], []
    width = max(len(r) for r in grid)
    grid = [list(r) + [None] * (width - len(r)) for r in grid]
    h = _detect_header(grid)
    if h is None:
        header = [f"kolumna_{j + 1}" for j in range(width)]
        start = 0
    else:
        header, start = [], h + 1
        used: dict[str, int] = {}
        for j, c in enumerate(grid[h]):
            name = _scalar_str(c) or f"kolumna_{j + 1}"
            if name in used:
                used[name] += 1
                name = f"{name}_{used[name]}"
            else:
                used[name] = 1
            header.append(name)
    rows, numbers = [], []
    for i in range(start, len(grid)):
        r = grid[i]
        if all(_is_empty(c) for c in r):
            continue
        rows.append({header[j]: r[j] for j in range(width) if not _is_empty(r[j])})
        numbers.append(i + 1)
    return header, rows, numbers


def _read_csv(path: Path) -> list[list[str]]:
    text = _read_text(path, limit=200_000_000)
    sep = "\t" if path.suffix.lower() == ".tsv" else _sniff_delimiter(text)
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    return [row for row in csv.reader(io.StringIO(text), delimiter=sep)]


def _records_from_json(data: Any, depth: int = 0) -> list[dict[str, Any]]:
    if isinstance(data, list):
        out = []
        for r in data:
            if isinstance(r, dict):
                out.append(r)
            elif isinstance(r, str) and r.strip():
                out.append(_scalar_record(r))
        return out
    if isinstance(data, str):
        return [_scalar_record(data)] if data.strip() else []
    if not isinstance(data, dict) or not data:
        return []
    lowered = {norm_key(k): v for k, v in data.items()}
    for key in (
        "records",
        "data",
        "items",
        "materials",
        "materialy",
        "ads",
        "reklamy",
        "rows",
        "results",
        "entries",
        "dane",
        "rekordy",
        "wyniki",
        "samples",
        "list",
    ):
        v = lowered.get(key)
        if isinstance(v, list) and all(isinstance(x, dict | str) for x in v):
            return _records_from_json(v, depth + 1)
        if isinstance(v, dict) and depth < 2:
            inner = _records_from_json(v, depth + 1)
            if len(inner) > 1:
                return inner
    values = list(data.values())
    if len(values) >= 2 and all(isinstance(v, dict) for v in values):  # słownik id → rekord
        k0, k1 = set(values[0]), set(values[1])
        if k0 and len(k0 & k1) / len(k0 | k1) >= 0.5:
            return [{"id": k, **v} if "id" not in v else v for k, v in data.items()]
    lists = [v for v in values if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
    if lists and depth == 0 and len(max(lists, key=len)) > 1:
        return max(lists, key=len)
    return [data]


def _scalar_record(s: str) -> dict[str, Any]:
    return {"url": s} if normalize_url(s) else {"text": s}


def _parse_jsonl(text: str, name: str) -> tuple[list[dict[str, Any]], list[int]]:
    rows, numbers, bad = [], [], 0
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(obj, dict):
            rows.append(obj)
            numbers.append(i)
        elif isinstance(obj, str) and obj.strip():
            rows.append(_scalar_record(obj))
            numbers.append(i)
    if bad:
        log.warning("%s: pominięto %d nieczytelnych wierszy JSONL", name, bad)
    return rows, numbers


_AUX_SHEETS = frozenset(
    {
        "legenda",
        "slownik",
        "info",
        "readme",
        "metadane",
        "metadata",
        "instrukcja",
        "schema",
        "summary",
        "podsumowanie",
        "opis_kolumn",
        "columns",
        "kolumny",
        "legend",
    }
)


@dataclass
class _Table:
    path: Path
    rows: list[dict[str, Any]]
    columns: list[str] | None = None
    numbers: list[int] | None = None
    sheet: str | None = None
    embedded: dict[int, list[Path]] | None = None


def _read_xlsx(path: Path, ctx: _Ctx) -> list[_Table]:
    from openpyxl import load_workbook

    wb = load_workbook(path, data_only=True)
    tables = []
    for ws in wb.worksheets:
        if len(wb.worksheets) > 1 and norm_key(ws.title) in _AUX_SHEETS:
            log.info("%s: pomijam arkusz pomocniczy „%s”", path.name, ws.title)
            continue
        grid: list[list[Any]] = []
        for row in ws.iter_rows(min_row=1, max_row=ws.max_row):
            vals = []
            for c in row:
                v = c.value
                link = getattr(getattr(c, "hyperlink", None), "target", None)
                if link and (v is None or not normalize_url(str(v))):
                    v = link
                vals.append(v)
            grid.append(vals)
        for rng in list(getattr(ws.merged_cells, "ranges", [])):  # scalone komórki (np. „Kampania” na kilka wierszy)
            top = grid[rng.min_row - 1][rng.min_col - 1] if rng.min_row - 1 < len(grid) else None
            for r in range(rng.min_row - 1, min(rng.max_row, len(grid))):
                for c in range(rng.min_col - 1, min(rng.max_col, len(grid[r]))):
                    grid[r][c] = top
        embedded: dict[int, list[Path]] = defaultdict(list)
        for img in getattr(ws, "_images", []):
            try:
                anchor = getattr(img.anchor, "_from", None)
                if anchor is None:
                    continue
                data = img._data()
                ext = "." + (img.format or "png").lower().replace("jpeg", "jpg")
                p = ctx.extractor.save(data, ext if ext in IMAGE_EXTS else ".png", f"{ws.title}-w{anchor.row + 1}")
                embedded[anchor.row + 1].append(p)
            except Exception as e:  # noqa: BLE001
                log.warning("%s: nie odczytano osadzonego obrazu: %s", path.name, e)
        columns, rows, numbers = _grid_to_rows(grid)
        if rows:
            tables.append(_Table(path, rows, columns, numbers, ws.title if len(wb.worksheets) > 1 else None, embedded))
    return tables


def _read_table(path: Path, ctx: _Ctx) -> list[_Table]:
    """Plik rekordów → tabele (XLSX: jedna na arkusz). Wyjątki przekazywane wyżej."""
    ext = path.suffix.lower()
    if ext in (".jsonl", ".ndjson"):
        rows, numbers = _parse_jsonl(_read_text(path, limit=2_000_000_000), path.name)
        return [_Table(path, rows, None, numbers)]
    if ext == ".json":
        text = _read_text(path, limit=2_000_000_000)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            rows, numbers = _parse_jsonl(text, path.name)  # JSONL z rozszerzeniem .json
            return [_Table(path, rows, None, numbers)]
        if isinstance(data, list) and len(data) > 1 and all(isinstance(r, list) for r in data):
            columns, rows, numbers = _grid_to_rows(data)  # tabela jako lista wierszy (pierwszy = nagłówek)
            return [_Table(path, rows, columns, numbers)]
        return [_Table(path, _records_from_json(data))]
    if ext in (".csv", ".tsv"):
        columns, rows, numbers = _grid_to_rows(_read_csv(path))
        return [_Table(path, rows, columns, numbers)]
    if ext in (".xlsx", ".xlsm"):
        return _read_xlsx(path, ctx)
    if ext == ".xls":
        import pandas as pd

        tables = []
        for name, df in pd.read_excel(path, sheet_name=None, header=None, dtype=object).items():
            columns, rows, numbers = _grid_to_rows(df.where(df.notna(), None).values.tolist())
            if rows:
                tables.append(_Table(path, rows, columns, numbers, str(name)))
        return tables
    raise ValueError(f"nieobsługiwany format pliku: {path.name}")


def _tables_to_drafts(tables: list[_Table], base: Path, ctx: _Ctx) -> list[_Draft]:
    out: list[_Draft] = []
    for t in tables:
        out.extend(
            _records_to_drafts(
                t.rows, t.columns, base, ctx, t.path, sheet=t.sheet, row_numbers=t.numbers, embedded=t.embedded
            )
        )
    return out


# ---------------------------------------------------------------- pojedyncze pliki (tryb katalogu)


def _kind(p: Path) -> str:
    ext = p.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in HTML_EXTS:
        return "html"
    if ext in MHTML_EXTS:
        return "mhtml"
    if ext in TEXT_EXTS:
        return "text"
    if ext in LINK_EXTS:
        return "link"
    if ext == ".json":
        return "json"
    if ext in TABLE_EXTS:
        return "table"
    return "other"


def _keep_file(name: str) -> bool:
    if name.startswith(".") or name.startswith("~$"):
        return False
    p = Path(name)
    if p.stem.lower() in IGNORED_STEMS:
        return False
    return _kind(p) != "other"


def _url_lines(text: str) -> tuple[list[str], int]:
    """(adresy, liczba niepustych wierszy) — pierwszy token wiersza wyglądający na URL."""
    urls, n = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        n += 1
        for tok in re.split(r"[\s,;]+", line):
            u = normalize_url(tok)
            if u:
                urls.append(u)
                break
    return urls, n


def _is_url_list(p: Path, text: str) -> bool:
    if p.suffix.lower() != ".txt":
        return False
    urls, n = _url_lines(text)
    return bool(urls) and (p.name.lower() in URL_LIST_NAMES or len(urls) / max(n, 1) >= 0.6)


def _apply_text_file(d: _Draft, p: Path) -> None:
    if p.suffix.lower() == ".eml":
        _apply_eml(d, p)
        return
    text = _read_text(p)
    if _is_url_list(p, text):
        for u in _url_lines(text)[0]:
            d.add_url(u)
    else:
        d.add_text(text.strip())


def _apply_eml(d: _Draft, p: Path) -> None:
    msg = email.message_from_bytes(p.read_bytes()[:MAX_TEXT_BYTES], policy=email.policy.default)
    parts = []
    subject = msg.get("subject")
    if subject:
        parts.append(str(subject))
    body = None
    try:
        body = msg.get_body(preferencelist=("plain", "html"))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        body = None
    if body is not None:
        try:
            content = body.get_content()
        except Exception:  # noqa: BLE001
            content = ""
        if body.get_content_type() == "text/html":
            content = _strip_markup(content)
        parts.append(str(content).strip())
    d.add_text("\n".join(x for x in parts if x).strip())
    sender = msg.get("from")
    if sender and d.advertiser is None:
        d.advertiser = str(sender)
    when = parse_date(msg.get("date")) if msg.get("date") else None
    if when is None and msg.get("date"):
        try:
            when = _naive_utc(email.utils.parsedate_to_datetime(str(msg.get("date"))))
        except (TypeError, ValueError):
            when = None
    if when and d.observed_at is None:
        d.observed_at = when


def _apply_mhtml(d: _Draft, p: Path, ctx: _Ctx) -> None:
    """Strona zapisana jako MHTML (Chrome „jeden plik”): część text/html → plik HTML, adres → url."""
    msg = email.message_from_bytes(p.read_bytes(), policy=email.policy.default)
    loc = msg.get("Snapshot-Content-Location")
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            payload = part.get_payload(decode=True)
            if isinstance(payload, bytes) and payload:
                d.html = ctx.extractor.save(payload, ".html", p.stem)
                loc = loc or part.get("Content-Location")
                break
    if loc:
        u = normalize_url(str(loc))
        if u:
            d.add_url(u)
    if d.html is None:
        d.warnings.append(f"nie znaleziono HTML w {p.name}")


def _apply_link_file(d: _Draft, p: Path) -> None:
    try:
        if p.suffix.lower() == ".webloc":
            data = plistlib.loads(p.read_bytes())
            target = data.get("URL") if isinstance(data, dict) else None
        else:
            m = re.search(r"^URL=(.+)$", _read_text(p), re.M)
            target = m.group(1).strip() if m else None
    except Exception as e:  # noqa: BLE001
        d.warnings.append(f"nieczytelny skrót {p.name}: {e}")
        return
    u = normalize_url(target) if target else None
    if u:
        d.add_url(u)


def _assemble(files: list[Path], base: Path, ctx: _Ctx, id_hint: str | None, rel_dir: str) -> _Draft:
    """Grupa plików (ta sama nazwa bazowa albo cały podkatalog) → jeden materiał."""
    d = _Draft()
    sidecars: list[dict[str, Any]] = []
    for f in files:
        k = _kind(f)
        try:
            if k == "image":
                d.add_image(f.resolve())
            elif k == "html":
                if d.html is None:
                    d.html = f.resolve()
                else:
                    d.meta.setdefault("extra_html", []).append(str(f.resolve()))
            elif k == "mhtml":
                if d.html is None:
                    _apply_mhtml(d, f, ctx)
            elif k == "text":
                _apply_text_file(d, f)
            elif k == "link":
                _apply_link_file(d, f)
            elif k in ("json", "table"):
                for t in _read_table(f, ctx):
                    if t.rows:
                        sidecars.append(t.rows[0])
                        break
        except Exception as e:  # noqa: BLE001
            log.warning("Nie wczytano pliku %s: %s", f, e)
            d.warnings.append(f"nie wczytano pliku {f.name}: {e}")
    for row in sidecars:
        flat = _flatten(row)
        _apply_row(d, flat, _plan_columns(list(flat), [flat]), ctx, base)
    if d.id is None:
        d.id = id_hint
    d.rel_dir = rel_dir
    d.meta.setdefault("source_path", rel_dir if len(files) != 1 else _rel(files[0], ctx.root))
    d.fingerprint = "|".join(sorted(f.name for f in files))
    return d


def _rel(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root).as_posix()
    except ValueError:
        return p.name


# ---------------------------------------------------------------- katalogi


@dataclass
class _Tree:
    files: dict[Path, list[Path]]
    subdirs: dict[Path, list[Path]]
    has_content: dict[Path, bool]


def _scan_tree(root: Path) -> _Tree:
    files: dict[Path, list[Path]] = {}
    subdirs: dict[Path, list[Path]] = {}
    order: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        depth = len(d.relative_to(root).parts)
        html_stems = {Path(f).stem.lower() for f in filenames if Path(f).suffix.lower() in HTML_EXTS}
        keep = []
        for n in sorted(dirnames):
            low = n.lower()
            if n.startswith(".") or low in SKIP_DIRS or (d / n).is_symlink() or depth >= MAX_DEPTH:
                continue
            if low.endswith(("_files", "_pliki")) and low[:-6] in html_stems:  # zasoby zapisanej strony
                continue
            keep.append(n)
        dirnames[:] = keep
        files[d] = [d / f for f in sorted(filenames) if _keep_file(f)]
        subdirs[d] = [d / n for n in keep]
        order.append(d)
    has: dict[Path, bool] = {}
    for d in reversed(order):
        has[d] = bool(files[d]) or any(has.get(s, False) for s in subdirs[d])
    return _Tree(files, subdirs, has)


def _best_table(paths: list[Path], ctx: _Ctx) -> tuple[list[_Table], int]:
    """Plik rekordów z największą liczbą wierszy (przy remisie: records.jsonl, data.json, …)."""
    preferred = ("records.jsonl", "data.json", "records.json", "data.jsonl", "dataset.jsonl", "dataset.json")
    best: tuple[list[_Table], int, int] | None = None
    for p in paths:
        try:
            tables = _read_table(p, ctx)
        except Exception as e:  # noqa: BLE001
            log.warning("Nie wczytano pliku rekordów %s: %s", p, e)
            continue
        n = sum(len(t.rows) for t in tables)
        pref = -preferred.index(p.name.lower()) if p.name.lower() in preferred else -99
        if best is None or (n, pref) > (best[1], best[2]):
            best = (tables, n, pref)
    return (best[0], best[1]) if best else ([], 0)


def _is_bundle(files: list[Path]) -> bool:
    if not files:
        return False
    if len(files) == 1:
        return True
    if len({f.stem.lower() for f in files}) == 1:
        return True
    kinds: dict[str, int] = defaultdict(int)
    for f in files:
        kinds[_kind(f)] += 1
    return len(kinds) >= 2 and len(files) <= 12 and kinds["image"] <= 4 and kinds["html"] + kinds["mhtml"] <= 2


def _load_dir(d: Path, root: Path, tree: _Tree, ctx: _Ctx, out: list[_Draft]) -> None:
    files = tree.files.get(d, [])
    table_files = [f for f in files if _kind(f) in ("json", "table")]
    content_subdirs = [s for s in tree.subdirs.get(d, []) if tree.has_content.get(s)]
    n = 0
    if table_files:
        tables, n = _best_table(table_files, ctx)
        others = [f for f in files if _kind(f) not in ("json", "table")]
        # plik rekordów: ≥ 2 wiersze i nie tylko jeden z wielu opisów-„sidecarów” (ad1.json, ad2.json, …)
        is_records = n >= 2 and n >= len(table_files) - 1
        if is_records or (n == 1 and not others and not (d != root and len(table_files) == 1)):
            log.info("Plik rekordów: %s (%d wierszy)", tables[0].path, n)
            out.extend(_tables_to_drafts(tables, d, ctx))
            return
    rel_dir = _rel(d, root) if d != root else ""
    # korzeń jako jeden materiał tylko z czytelnym, jednowierszowym opisem (info.json + zrzut + strona)
    root_bundle = len(table_files) == 1 and n == 1 and len(files) > 1
    if not content_subdirs and _is_bundle(files) and (d != root or root_bundle):
        out.append(_assemble(files, d, ctx, rel_dir or d.name, rel_dir))
        return
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        groups[_base_stem(f).lower()].append(f)
    for _stem, group in groups.items():
        url_lists = []
        rest = []
        for f in group:
            if f.suffix.lower() == ".txt":
                try:
                    if _is_url_list(f, _read_text(f)):
                        url_lists.append(f)
                        continue
                except OSError as e:
                    log.warning("Nie wczytano %s: %s", f, e)
                    continue
            rest.append(f)
        for f in url_lists:
            for i, u in enumerate(_url_lines(_read_text(f))[0], 1):
                dd = _Draft(urls=[u], rel_dir=rel_dir)
                dd.meta.update({"source_file": _rel(f, root), "source_row": i})
                out.append(dd)
        if not rest:
            continue
        stem = _base_stem(min(rest, key=lambda f: len(f.name)))
        stem_id = f"{rel_dir}/{stem}" if rel_dir else stem
        if all(_kind(f) in ("json", "table") for f in rest):
            for f in rest:
                try:
                    drafts = _tables_to_drafts(_read_table(f, ctx), d, ctx)
                except Exception as e:  # noqa: BLE001
                    log.warning("Nie wczytano %s: %s", f, e)
                    continue
                for dd in drafts:
                    dd.rel_dir = rel_dir
                    if dd.id is None and len(drafts) == 1:
                        dd.id = stem_id
                out.extend(drafts)
            continue
        out.append(_assemble(rest, d, ctx, stem_id, rel_dir))
    for s in content_subdirs:
        _load_dir(s, root, tree, ctx, out)


def _base_stem(f: Path) -> str:
    """Nazwa bez wszystkich znanych rozszerzeń: ``ad1.png.json`` → ``ad1`` (opisy w stylu Google Takeout)."""
    known = IMAGE_EXTS | HTML_EXTS | MHTML_EXTS | TEXT_EXTS | LINK_EXTS | TABLE_EXTS
    name = f.name
    while Path(name).suffix.lower() in known and Path(name).stem:
        name = Path(name).stem
    return name


def _load_file(p: Path, ctx: _Ctx) -> list[_Draft]:
    k = _kind(p)
    if k in ("json", "table"):
        try:
            tables = _read_table(p, ctx)
        except Exception as e:  # noqa: BLE001 — czytelny komunikat zamiast śladu stosu
            raise ValueError(f"Nie udało się wczytać pliku {p.name}: {e}") from e
        return _tables_to_drafts(tables, p.parent, ctx)
    if k == "other":
        raise ValueError(f"Nieobsługiwany format pliku: {p.name}")
    if k == "text" and p.suffix.lower() == ".txt":
        text = _read_text(p)
        if _is_url_list(p, text):
            drafts = []
            for i, u in enumerate(_url_lines(text)[0], 1):
                dd = _Draft(urls=[u])
                dd.meta.update({"source_file": p.name, "source_row": i})
                drafts.append(dd)
            return drafts
    return [_assemble([p], p.parent, ctx, p.stem, "")]


# ---------------------------------------------------------------- finalizacja


def _first_bytes(paths: Iterable[Path | str]) -> bytes:
    for p in paths:
        try:
            return Path(p).read_bytes()
        except OSError:
            continue
    return b""


def _content_hash(url: str | None, text: str | None, images: list[Path], html: Path | None, fallback: str) -> str:
    h = hashlib.sha1()
    h.update((url or "").encode("utf-8"))
    h.update(b"|")
    h.update((text or "").encode("utf-8"))
    h.update(b"|")
    data = _first_bytes(images)
    if not data and html is not None:
        data = _first_bytes([html])
    h.update(data)
    if not (url or text or data):
        h.update(fallback.encode("utf-8"))
    return h.hexdigest()


def _source_kind(url: str | None, text: str | None, images: list[Any], html: Any) -> SourceKind:
    kinds = [
        k
        for k, present in (
            (SourceKind.URL, bool(url)),
            (SourceKind.TEXT, bool(text)),
            (SourceKind.IMAGE, bool(images)),
            (SourceKind.HTML, bool(html)),
        )
        if present
    ]
    return SourceKind.RECORD if len(kinds) != 1 else kinds[0]


def _to_material(d: _Draft, mid: str) -> Material:
    if not d.urls and d.domains:
        for dom in d.domains:
            u = normalize_url(dom)
            if u:
                d.add_url(u)
                break
    url = d.urls[0] if d.urls else None
    text = "\n".join(t for t in d.texts if t).strip() or None
    images = list(dict.fromkeys(str(Path(p).resolve()) for p in d.images))
    html = str(Path(d.html).resolve()) if d.html else None
    meta = dict(d.meta)
    if d.urls[1:]:
        meta["urls"] = d.urls[1:]
    if d.image_urls:
        meta["image_urls"] = list(dict.fromkeys(d.image_urls))
    infra = {**_coerce_infra(d.infra_cols), **_coerce_infra(d.infra)} if (d.infra or d.infra_cols) else {}
    if infra:
        meta["infra"] = infra
    if d.warnings:
        meta["ingest_warnings"] = list(dict.fromkeys(d.warnings))
    if not (url or text or images or html):
        meta.setdefault("ingest_warnings", []).append("materiał bez treści (brak URL, tekstu, obrazu i HTML)")
    return Material(
        id=mid,
        source=_source_kind(url, text, images, html),
        url=url,
        text=text,
        image_paths=images,
        html_path=html,
        platform=d.platform,
        advertiser=d.advertiser,
        observed_at=d.observed_at,
        label=d.label,
        meta=_json_safe(meta),
    )


def _finalize(drafts: list[_Draft], *, label_from_dir: bool = False) -> list[Material]:
    given = {d.id for d in drafts if d.id}
    used: set[str] = set()
    out: list[Material] = []
    for d in drafts:
        try:
            if label_from_dir and d.label is None and d.rel_dir:
                d.label = d.rel_dir.split("/")[0]
            mid = d.id
            if not mid:
                url = d.urls[0] if d.urls else None
                text = "\n".join(t for t in d.texts if t).strip() or None
                mid = "m-" + _content_hash(url, text, d.images, d.html, d.fingerprint)[:12]
            if mid in used:
                k = 2
                while f"{mid}-{k}" in used or f"{mid}-{k}" in given:
                    k += 1
                mid = f"{mid}-{k}"
            used.add(mid)
            out.append(_to_material(d, mid))
        except Exception as e:  # noqa: BLE001
            log.warning("Pominięto materiał %s: %s", d.id or "?", e)
    return out


# ---------------------------------------------------------------- API


def load_dataset(path: Path | str, *, label_from_dir: bool = False) -> list[Material]:
    """Wczytuje zbiór w dowolnym obsługiwanym formacie → lista materiałów (ścieżki absolutne).

    ``label_from_dir=True`` — materiały z podkatalogów bez etykiety dostają nazwę podkatalogu pierwszego
    poziomu jako etykietę (zbiory ułożone „kampania/plik”; tylko do ewaluacji).
    Nieistniejąca ścieżka → ``FileNotFoundError``; zły rekord/plik wewnątrz zbioru jest pomijany z ostrzeżeniem.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Nie znaleziono zbioru danych: {p}")
    p = p.resolve()
    ctx = _Ctx(root=p if p.is_dir() else p.parent, extractor=_Extractor(p))
    drafts: list[_Draft] = []
    if p.is_dir():
        _load_dir(p, p, _scan_tree(p), ctx, drafts)
    else:
        drafts = _load_file(p, ctx)
    materials = _finalize(drafts, label_from_dir=label_from_dir)
    log.info("Wczytano %d materiałów z %s", len(materials), p)
    return materials


def material_from_input(
    url: str | None = None,
    image_path: str | Path | list[str | Path] | None = None,
    html_path: str | Path | None = None,
    text: str | None = None,
) -> Material:
    """Nowy materiał z formularza / CLI (id ``new-`` + skrót treści). Nigdy nie rzuca wyjątku."""
    d = _Draft()
    raw_url = str(url).strip() if url else ""
    if raw_url:
        u = normalize_url(raw_url)
        if u:
            d.add_url(u)
        else:
            d.add_url(raw_url)
            d.warnings.append("podany adres nie wygląda na poprawny URL")
    paths = image_path if isinstance(image_path, list | tuple) else ([image_path] if image_path else [])
    for ip in paths:
        p = Path(ip).expanduser().resolve()
        if not p.is_file():
            d.warnings.append(f"brak pliku obrazu: {p}")
        d.add_image(p)
    if html_path:
        hp = Path(html_path).expanduser().resolve()
        if not hp.is_file():
            d.warnings.append(f"brak pliku HTML: {hp}")
        d.html = hp
    if text and str(text).strip():
        d.add_text(str(text).strip())
    if not (d.urls or d.images or d.html or d.texts):
        d.warnings.append("brak danych wejściowych")
    first_url = d.urls[0] if d.urls else None
    joined = "\n".join(d.texts) or None
    h = _content_hash(first_url, joined, d.images, d.html, "pusty")
    return _to_material(d, "new-" + h[:10])
