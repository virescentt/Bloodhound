"""Analiza HTML strony (bez wykonywania JavaScriptu): tytuł, opis, widoczny tekst, identyfikatory trackerów,
domeny skryptów, formularze, linki wychodzące, favicon, odcisk szablonu DOM (SimHash) i skrót treści.

Nigdy nie rzuca wyjątku: zepsuty HTML daje częściowy wynik (każda cecha liczona osobno).

Identyfikatory trackerów (``tracker_ids``) mają postać ``<rodzaj>:<id>``:
``ga:UA-1234-1``, ``gtag:G-XXXX``, ``gtm:GTM-XXXX``, ``aw:AW-123``, ``fbpixel:123…``, ``ym:123``, ``tiktok:XXXX``,
``hotjar:123``, ``clarity:abc``, ``adsense:ca-pub-…``, weryfikacje domen (``gsv:…``, ``fbdv:…``), ukryte pola
formularzy z identyfikatorem oferty/strumienia (``form:offer_id=123``) oraz odciski platform trackerów kliknięć
(``tracker:keitaro``, ``tracker:binom``, ``tracker:voluum``, ``tracker:redtrack``) — te ostatnie to rodzaj
oprogramowania, a NIE identyfikator operatora (słaba przesłanka). Opisy po polsku: ``describe_tracker``.
"""

from __future__ import annotations

import base64
import codecs
import hashlib
import logging
import math
import re
import warnings
from collections import Counter
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

import numpy as np
from bs4 import BeautifulSoup, Tag

from slad.features.infra.domains import host_of, normalize_url, registered_domain, unique

try:  # ostrzeżenia bs4 o „tekście podobnym do adresu/pliku” i XHTML — bez znaczenia dla nas
    from bs4 import MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:  # pragma: no cover
    pass

log = logging.getLogger(__name__)

TEXT_LIMIT = 20_000  # znaków widocznego tekstu
SIMHASH_DEPTH = 6  # ścieżka DOM = ostatnie N znaczników (odporność na dodatkowe opakowania)
SIMHASH_MIN_ELEMENTS = 12  # mniej elementów → strona-zaślepka, odcisk szablonu bez znaczenia
JS_REDIRECT_MAX_CHARS = 30_000  # przekierowania JS śledzimy tylko na małych stronach
LIST_CAP = 200

# ---------------------------------------------------------------- trackery

_TRACKER_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ga", re.compile(r"(?<![A-Za-z0-9])(UA-\d{4,10}-\d{1,4})(?!\d)")),
    ("gtm", re.compile(r"(?<![A-Za-z0-9])(GTM-[A-Z0-9]{4,9})(?![A-Za-z0-9])")),
    ("gtag", re.compile(r"(?<![A-Za-z0-9_-])((?:G|GT)-[A-Z0-9]{6,12})(?![A-Za-z0-9_-])")),
    ("aw", re.compile(r"(?<![A-Za-z0-9_-])(AW-\d{6,12})(?!\d)")),
    ("dc", re.compile(r"(?<![A-Za-z0-9_-])(DC-\d{6,10})(?!\d)")),
    ("fbpixel", re.compile(r"fbq\(\s*['\"]init['\"]\s*,\s*['\"]?(\d{8,20})")),
    ("fbpixel", re.compile(r"facebook\.com/tr/?\?(?:[^\"'\s<>]*?(?:&amp;|&))?id=(\d{8,20})")),
    ("fbpixel", re.compile(r"connect\.facebook\.net/signals/config/(\d{8,20})")),
    (
        "fbpixel",
        re.compile(r"(?i)\b(?:fb_?pixel(?:_?id)?|pixel_?id|facebook_?pixel(?:_?id)?)['\"]?\s*[:=]\s*['\"]?(\d{12,20})"),
    ),
    ("ym", re.compile(r"\bym\(\s*['\"]?(\d{5,12})['\"]?\s*,\s*['\"]init['\"]")),
    ("ym", re.compile(r"mc\.yandex\.[a-z.]{2,6}/watch/(\d{5,12})")),
    ("ym", re.compile(r"\byaCounter(\d{5,12})")),
    ("ym", re.compile(r"Ya\.Metrika2?\(\s*\{\s*id\s*:\s*(\d{5,12})")),
    ("tiktok", re.compile(r"ttq\.load\(\s*['\"]([A-Z0-9]{15,25})['\"]")),
    (
        "tiktok",
        re.compile(
            r"analytics\.tiktok\.com/i18n/pixel/(?:events|sdk)\.js\?(?:[^\"'\s<>]*?(?:&amp;|&))?sdkid=([A-Z0-9]{15,25})"
        ),
    ),
    ("hotjar", re.compile(r"\bhjid\s*:\s*(\d{5,10})")),
    ("hotjar", re.compile(r"static\.hotjar\.com/c/hotjar-(\d{5,10})\.js")),
    ("clarity", re.compile(r"clarity\.ms/tag/([a-z0-9]{8,12})")),
    ("clarity", re.compile(r"['\"]clarity['\"]\s*,\s*['\"]script['\"]\s*,\s*['\"]([a-z0-9]{8,12})['\"]")),
    ("adsense", re.compile(r"(?<![A-Za-z0-9-])(ca-pub-\d{10,20})(?!\d)")),
    ("snap", re.compile(r"snaptr\(\s*['\"]init['\"]\s*,\s*['\"]([0-9a-f-]{36})['\"]")),
    ("twitter", re.compile(r"twq\(\s*['\"](?:init|config)['\"]\s*,\s*['\"]([a-z0-9]{5,10})['\"]")),
    ("pinterest", re.compile(r"pintrk\(\s*['\"]load['\"]\s*,\s*['\"](\d{10,16})['\"]")),
    ("linkedin", re.compile(r"_linkedin_partner_id\s*=\s*['\"](\d{4,10})['\"]")),
    ("vk", re.compile(r"VK\.Retargeting\.Init\(\s*['\"]([A-Za-z0-9-]{5,40})['\"]")),
)

# Odciski platform trackerów kliknięć (rodzaj oprogramowania, nie identyfikator operatora).
_CLICK_TRACKER_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "keitaro",
        re.compile(
            r"(?i)keitaro|\bKTracking\b|[?&](?:amp;)?_lp=1\b[^\"'\s<>]*?_token=|[?&](?:amp;)?_token=[^&\"'\s<>]+&(?:amp;)?_lp=1\b"
        ),
    ),
    ("binom", re.compile(r"(?i)\bbinom\b|/click\.php\?(?:[^\"'\s<>]*?&(?:amp;)?)?lp=")),
    ("voluum", re.compile(r"(?i)voluum|volumtrk")),
    ("redtrack", re.compile(r"(?i)redtrack|\brdtk\.io\b|rtkclickid|\brtkcid\b")),
    ("bemob", re.compile(r"(?i)\bbemob")),
)

# Identyfikator piksela przekazywany w adresie strony (?pixel=…) — częsty w kampaniach na Facebooku.
_URL_PIXEL_RE = re.compile(
    r"(?i)(?:^|[?&;])(?:pixel|fbpixel|fb_pixel|pixel_id|fbpx|fb_px|px|fb)=(\d{12,20})(?=$|[&#;])"
)
_URL_TT_PIXEL_RE = re.compile(r"(?i)(?:^|[?&;])(?:ttpixel|tt_pixel|tiktok_pixel|ttpx)=([A-Z0-9]{15,25})(?=$|[&#;])")

_VERIFY_META = {
    "google-site-verification": "gsv",
    "facebook-domain-verification": "fbdv",
    "yandex-verification": "yandexv",
    "msvalidate.01": "bingv",
    "p:domain_verify": "pinv",
}

# Ukryte pola formularza z identyfikatorem oferty / strumienia / partnera (sieci CPA) — wspólne dla kampanii.
_CAMPAIGN_FIELD_RE = re.compile(
    r"(?i)offer|flow|stream|campaign|landing|lander|^aff|_aff|partner|webmaster|publisher|^wm_?id$|^pid$|^lp_?id$|goal"
)
_MACRO_RE = re.compile(r"^\s*(?:\{.*\}|\[.*\]|%.*%|\$\{.*\}|<.*>)\s*$")

TRACKER_LABELS: dict[str, str] = {
    "ga": "Google Analytics (UA)",
    "gtag": "Google Analytics 4 / tag Google",
    "gtm": "Google Tag Manager",
    "aw": "Google Ads",
    "dc": "Google Floodlight",
    "fbpixel": "Facebook Pixel",
    "ym": "Yandex Metrica",
    "tiktok": "TikTok Pixel",
    "hotjar": "Hotjar",
    "clarity": "Microsoft Clarity",
    "adsense": "Google AdSense",
    "snap": "Snapchat Pixel",
    "twitter": "X/Twitter Pixel",
    "pinterest": "Pinterest Tag",
    "linkedin": "LinkedIn Insight",
    "vk": "VK Pixel",
    "gsv": "weryfikacja Google Search Console",
    "fbdv": "weryfikacja domeny Facebook Business",
    "yandexv": "weryfikacja Yandex Webmaster",
    "bingv": "weryfikacja Bing Webmaster",
    "pinv": "weryfikacja domeny Pinterest",
    "form": "ukryte pole formularza (oferta/strumień)",
    "tracker": "platforma trackera kliknięć",
}

_KNOWN_PREFIXES = frozenset(TRACKER_LABELS)


def describe_tracker(value: str) -> str:
    """``fbpixel:123`` → ``Facebook Pixel 123`` (opis po polsku do UI i przesłanek)."""
    kind, _, ident = (value or "").partition(":")
    label = TRACKER_LABELS.get(kind.lower())
    if not label or not ident:
        return value
    if kind.lower() == "tracker":
        return f"{label} {ident.capitalize()}"
    return f"{label} {ident}"


def normalize_tracker_id(value: Any, hint: str | None = None) -> str | None:
    """Ujednolica identyfikator trackera z danych wejściowych (np. ``UA-1-1`` → ``ga:UA-1-1``).
    ``hint`` — rodzaj z nazwy pola (np. ``fbpixel`` dla kolumny „fb_pixel”)."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if ":" in s:
        kind, _, ident = s.partition(":")
        if kind.lower() in _KNOWN_PREFIXES and ident.strip():
            return f"{kind.lower()}:{ident.strip()}"
    up = s.upper()
    if re.fullmatch(r"UA-\d{4,10}-\d{1,4}", up):
        return f"ga:{up}"
    if re.fullmatch(r"GTM-[A-Z0-9]{4,9}", up):
        return f"gtm:{up}"
    if re.fullmatch(r"(?:G|GT)-[A-Z0-9]{6,12}", up):
        return f"gtag:{up}"
    if re.fullmatch(r"AW-\d{6,12}", up):
        return f"aw:{up}"
    if re.fullmatch(r"CA-PUB-\d{10,20}", up):
        return f"adsense:{s.lower()}"
    if hint and hint.lower() in _KNOWN_PREFIXES:
        return f"{hint.lower()}:{s}"
    return s


def extract_tracker_ids(code: str) -> list[str]:
    """Identyfikatory trackerów z kodu strony (skrypty, atrybuty, adresy)."""
    out: list[str] = []
    if not code:
        return out
    for kind, rx in _TRACKER_RULES:
        for m in rx.finditer(code):
            ident = m.group(1)
            if kind in ("ga", "gtm", "gtag", "aw", "dc", "tiktok"):
                ident = ident.upper()
            out.append(f"{kind}:{ident}")
    return unique(out)


def click_tracker_ids(*texts: str | None) -> list[str]:
    """Odciski platform trackerów kliknięć (Keitaro, Binom, Voluum, RedTrack, BeMob) w kodzie lub adresach."""
    out: list[str] = []
    for text in texts:
        if not text:
            continue
        for name, rx in _CLICK_TRACKER_RULES:
            if rx.search(text):
                out.append(f"tracker:{name}")
    return unique(out)


def url_tracker_ids(urls: list[str] | tuple[str, ...]) -> list[str]:
    """Identyfikatory pikseli przekazywane w adresach (``?pixel=…``) + odciski trackerów kliknięć w adresach."""
    out: list[str] = []
    for u in urls:
        if not u:
            continue
        try:
            query = urlsplit(u).query
        except ValueError:
            query = ""
        if query:
            out += [f"fbpixel:{m.group(1)}" for m in _URL_PIXEL_RE.finditer(query)]
            out += [f"tiktok:{m.group(1).upper()}" for m in _URL_TT_PIXEL_RE.finditer(query)]
    out += click_tracker_ids(*urls)
    return unique(out)


# ---------------------------------------------------------------- dekodowanie

_META_CHARSET_RE = re.compile(rb"""<meta[^>]{0,200}?charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]{2,40})""", re.I)
_HEADER_CHARSET_RE = re.compile(r"""charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]{2,40})""", re.I)


def decode_html_bytes(data: bytes, content_type: str | None = None) -> str:
    """Bajty strony → tekst: BOM, charset z nagłówka, z ``<meta charset>``, awaryjnie UTF-8 z zamianą błędnych znaków."""
    if not data:
        return ""
    if data.startswith(codecs.BOM_UTF8):
        return data[3:].decode("utf-8", errors="replace")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    candidates: list[str] = []
    if content_type:
        m = _HEADER_CHARSET_RE.search(content_type)
        if m:
            candidates.append(m.group(1))
    m2 = _META_CHARSET_RE.search(data[:8192])
    if m2:
        candidates.append(m2.group(1).decode("ascii", "ignore"))
    for enc in candidates:
        try:
            codecs.lookup(enc)
        except LookupError:
            continue
        try:
            return data.decode(enc, errors="replace")
        except Exception:  # noqa: BLE001
            continue
    return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------- przekierowania po stronie klienta

_META_TAG_RE = re.compile(r"<meta\b[^>]*>", re.I)
_REFRESH_RE = re.compile(r"""http-equiv\s*=\s*["']?\s*refresh""", re.I)
_CONTENT_ATTR_RE = re.compile(r"""\bcontent\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.I)
_NOSCRIPT_RE = re.compile(r"<noscript\b.*?</noscript\s*>", re.I | re.S)
_SCRIPT_RE = re.compile(r"<script\b([^>]*)>(.*?)</script\s*>", re.I | re.S)
_LOC = r"(?:\b(?:window|document|top|self|parent)\s*\.\s*)?\blocation"
_JS_REDIRECT_RES: tuple[re.Pattern[str], ...] = (
    re.compile(_LOC + r"\s*\.\s*(?:replace|assign)\s*\(\s*([\"'`])((?:(?!\1).){1,2000})\1\s*\)"),
    re.compile(_LOC + r"(?:\s*\.\s*href)?\s*=\s*([\"'`])((?:(?!\1).){1,2000})\1"),
)
_JS_ATOB_RES: tuple[re.Pattern[str], ...] = (
    re.compile(
        _LOC + r"\s*\.\s*(?:replace|assign)\s*\(\s*(?:window\.)?atob\(\s*['\"]([A-Za-z0-9+/=]{8,4000})['\"]\s*\)"
    ),
    re.compile(_LOC + r"(?:\s*\.\s*href)?\s*=\s*(?:window\.)?atob\(\s*['\"]([A-Za-z0-9+/=]{8,4000})['\"]\s*\)"),
)
_JS_ESCAPE_RE = re.compile(r"\\(?:u([0-9a-fA-F]{4})|x([0-9a-fA-F]{2})|(/))")
_SCHEME_PREFIX_RE = re.compile(r"^[a-z][a-z0-9+.\-]*:", re.I)


def _abs_url(u: str | None, base: str | None) -> str | None:
    """Adres bezwzględny http(s) (względny → względem ``base``); inne schematy i brak bazy → None."""
    if not u:
        return None
    u = u.strip().replace("\n", "").replace("\r", "").replace("\t", "")
    if not u:
        return None
    low = u.lower()
    if low.startswith(("http://", "https://")):
        return u
    if low.startswith("//"):
        return ("https:" if base and base.lower().startswith("https:") else "http:") + u
    if _SCHEME_PREFIX_RE.match(low):
        return None
    if not base:
        return None
    try:
        out = urljoin(base, u)
    except ValueError:
        return None
    return out if out.lower().startswith(("http://", "https://")) else None


def _js_unescape(s: str) -> str:
    def repl(m: re.Match[str]) -> str:
        if m.group(1):
            return chr(int(m.group(1), 16))
        if m.group(2):
            return chr(int(m.group(2), 16))
        return "/"

    return _JS_ESCAPE_RE.sub(repl, s)


def parse_refresh(content: str) -> tuple[float, str | None]:
    """Wartość ``<meta http-equiv=refresh content=…>`` → (opóźnienie w s, adres lub None)."""
    m = re.match(r"\s*(\d+(?:\.\d+)?)?\s*[;,]?\s*(?:url\s*=\s*)?(.*)$", content or "", re.I | re.S)
    if not m:
        return 0.0, None
    delay = float(m.group(1)) if m.group(1) else 0.0
    target = (m.group(2) or "").strip().strip("'\"").strip()
    return delay, (target or None)


def _same_page(a: str, b: str | None) -> bool:
    if not b:
        return False
    return (normalize_url(a) or a) == (normalize_url(b) or b)


def client_redirect(
    html: str | None, base_url: str | None = None, *, allow_js: bool = True, max_delay: float = 10.0
) -> tuple[str, str] | None:
    """Przekierowanie po stronie klienta: ``<meta http-equiv=refresh>`` (opóźnienie ≤ ``max_delay`` s, poza
    ``<noscript>``) albo proste przekierowanie JS w skrypcie (``window.location(.href) = "…"``,
    ``location.replace/assign("…")``, ``top.location``, także ``atob("…")``) — tylko na małych stronach.
    Zwraca (adres bezwzględny, "meta" | "js") albo None. Kodu JS nie wykonujemy."""
    if not html:
        return None
    try:
        head = _NOSCRIPT_RE.sub(" ", html[:200_000])
        for tag in _META_TAG_RE.findall(head):
            if not _REFRESH_RE.search(tag):
                continue
            m = _CONTENT_ATTR_RE.search(tag)
            if not m:
                continue
            delay, target = parse_refresh(next((g for g in m.groups() if g is not None), ""))
            if target is None or delay > max_delay:
                continue
            url = _abs_url(target, base_url)
            if url and not _same_page(url, base_url):
                return url, "meta"
        if not allow_js or len(html) > JS_REDIRECT_MAX_CHARS:
            return None
        for attrs, body in _SCRIPT_RE.findall(head):
            if re.search(r"\bsrc\s*=", attrs, re.I) or not body.strip():
                continue
            for rx in _JS_ATOB_RES:
                for m in rx.finditer(body):
                    try:
                        decoded = base64.b64decode(m.group(1) + "===", validate=False).decode("utf-8", "strict")
                    except Exception:  # noqa: BLE001
                        continue
                    url = _abs_url(decoded, base_url)
                    if url and not _same_page(url, base_url):
                        return url, "js"
            for rx in _JS_REDIRECT_RES:
                for m in rx.finditer(body):
                    target = _js_unescape(m.group(2)).strip()
                    if not target or "${" in target or target.startswith("#"):
                        continue
                    url = _abs_url(target, base_url)
                    if url and host_of(url) and not _same_page(url, base_url):
                        return url, "js"
    except Exception as e:  # noqa: BLE001
        log.debug("client_redirect: %s", e)
    return None


# ---------------------------------------------------------------- strony „ogólne” (parking, zawieszone, błędy)

_GENERIC_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "parking domeny",
        re.compile(
            r"(?i)domain\s+(?:name\s+)?(?:is|may\s+be)\s+for\s+sale|this\s+domain\s+(?:is\s+)?(?:parked|registered\s+at|available)"
            r"|parked\s+(?:free|domain|by)|buy\s+this\s+domain|domain\s+has\s+expired|sedoparking|parkingcrew|bodis\.com"
            r"|\bdan\.com\b|afternic|hugedomains|domena\s+(?:jest\s+)?na\s+sprzeda|domena\s+(?:została\s+)?zaparkowana"
            r"|aftermarket\.pl"
        ),
    ),
    (
        "konto hostingu zawieszone",
        re.compile(
            r"(?i)account\s+(?:has\s+been\s+)?suspended|suspendedpage\.cgi|this\s+(?:site|website|account)\s+(?:has\s+been\s+|is\s+)?suspended"
            r"|(?:konto|strona)\s+(?:zostało\s+|została\s+)?zawieszon|website\s+is\s+(?:currently\s+)?unavailable"
        ),
    ),
    (
        "domyślna strona serwera",
        re.compile(
            r"(?i)welcome\s+to\s+nginx|apache2?\s+(?:ubuntu\s+|debian\s+)?default\s+page|\bit\s+works!|default\s+web\s+site\s+page"
            r"|iis\s+windows\s+server|^\s*index\s+of\s+/|congratulations!?\s+your\s+(?:website|site)\s+is\s+(?:up|live)"
            r"|future\s+home\s+of\s+something\s+quite\s+cool|domain\s+(?:is\s+)?not\s+configured|under\s+construction"
            r"|coming\s+soon"
        ),
    ),
    (
        "ostrzeżenie o phishingu",
        re.compile(
            r"(?i)suspected\s+phishing|deceptive\s+site\s+ahead|phishing\s+site\s+ahead|reported\s+as\s+(?:a\s+)?phishing"
        ),
    ),
    (
        "ochrona przed botami",
        re.compile(
            r"(?i)just\s+a\s+moment\.\.\.|attention\s+required!?\s*\|\s*cloudflare|checking\s+your\s+browser"
            r"|ddos\s+protection\s+by|enable\s+javascript\s+and\s+cookies\s+to\s+continue|verify(?:ing)?\s+you\s+are\s+(?:a\s+)?human"
        ),
    ),
    (
        "strona nie istnieje",
        re.compile(
            r"(?i)there\s+isn't\s+a\s+github\s+pages\s+site\s+here|\bsite\s+not\s+found\b|\b404\b.{0,20}not\s+found"
            r"|\bpage\s+not\s+found\b|deployment\s+(?:not\s+found|has\s+been\s+disabled)|no\s+such\s+app"
            r"|the\s+requested\s+url\s+was\s+not\s+found|nie\s+znaleziono\s+strony|strona\s+nie\s+(?:istnieje|została\s+znaleziona)"
        ),
    ),
)
_GENERIC_TEXT_MAX = 2_000  # tylko krótkie strony mogą być „ogólne” (parking, zawieszenie, błąd)


def _detect_generic(title: str | None, text: str, raw: str) -> str | None:
    short = len(text) < _GENERIC_TEXT_MAX
    probe = " ".join(x for x in ((title or ""), text[:3000] if short else "") if x)
    for reason, rx in _GENERIC_RULES:
        if probe and rx.search(probe):
            return reason
    if short and re.search(
        r"(?i)sedoparking|parkingcrew|bodis\.com|suspendedpage\.cgi|/cdn-cgi/challenge-platform", raw
    ):
        return "parking domeny lub ochrona przed botami"
    return None


def generic_page_reason(parsed: dict[str, Any] | None, status: int | None = None) -> str | None:
    """Powód, dla którego strona jest „ogólna” (błąd HTTP, parking, zawieszone konto, ochrona przed botami…) —
    jej szablon, favicon, tytuł i trackery należą do hostingu, nie do kampanii. None — zwykła strona."""
    if status is not None and status >= 400:
        return f"błąd HTTP {status}"
    if parsed:
        return parsed.get("generic")
    return None


# ---------------------------------------------------------------- SimHash szablonu DOM

_DOM_SKIP = frozenset({"script", "style", "noscript", "template", "meta", "link", "base", "br", "wbr", "title", "head"})
_DOM_LEAF = frozenset({"svg", "math", "iframe", "object", "video", "audio", "picture", "canvas", "select"})
_BIT_SHIFTS = np.arange(64, dtype=np.uint64)


def _dom_paths(soup: Any, limit: int = 30_000) -> list[str]:
    """Ścieżki znaczników w kolejności dokumentu, np. ``body>div>form>input:tel`` (ostatnie SIMHASH_DEPTH)."""
    out: list[str] = []
    stack: list[tuple[Any, tuple[str, ...]]] = []

    def push_children(node: Any, path: tuple[str, ...]) -> None:
        kids: list[tuple[Any, tuple[str, ...]]] = []
        for child in getattr(node, "children", ()):
            if not isinstance(child, Tag):
                continue
            name = (child.name or "").lower()
            if not name or name in _DOM_SKIP or ":" in name:
                continue
            tok = name
            if name == "input":
                itype = child.get("type")
                tok = f"input:{(itype if isinstance(itype, str) else 'text').strip().lower()[:12] or 'text'}"
            kids.append((child, (path + (tok,))[-SIMHASH_DEPTH:]))
        stack.extend(reversed(kids))

    push_children(soup, ())
    while stack and len(out) < limit:
        node, path = stack.pop()
        out.append(">".join(path))
        if (node.name or "").lower() not in _DOM_LEAF:
            push_children(node, path)
    return out


def _simhash(weights: dict[str, float]) -> int:
    feats = list(weights)
    hashes = np.fromiter(
        (
            int.from_bytes(hashlib.blake2b(f.encode("utf-8", "replace"), digest_size=8).digest(), "little")
            for f in feats
        ),
        dtype=np.uint64,
        count=len(feats),
    )
    w = np.fromiter((weights[f] for f in feats), dtype=np.float64, count=len(feats))
    bits = ((hashes[:, None] >> _BIT_SHIFTS) & np.uint64(1)).astype(np.float64)  # n × 64
    v = (bits * 2.0 - 1.0).T @ w
    fp = 0
    for i in range(64):
        if v[i] > 0:
            fp |= 1 << i
    return fp


def dom_simhash_from_paths(paths: list[str]) -> str | None:
    """64-bitowy SimHash (hex) struktury strony: cechy = ścieżki znaczników i pary kolejnych ścieżek, waga
    1 + log2(liczności). Ten sam szablon z innymi tekstami → niemal identyczny skrót (odległość Hamminga ≤ 3)."""
    if len(paths) < SIMHASH_MIN_ELEMENTS:
        return None
    counts: Counter[str] = Counter()
    prev = "^"
    for p in paths:
        counts["u:" + p] += 1
        counts["b:" + prev + "|" + p] += 1
        prev = p
    weights = {f: 1.0 + math.log2(c) for f, c in counts.items()}
    return f"{_simhash(weights):016x}"


def dom_simhash(html: str) -> str | None:
    soup = _make_soup(html)
    return dom_simhash_from_paths(_dom_paths(soup)) if soup is not None else None


def simhash_distance(a: str | None, b: str | None) -> int:
    """Odległość Hamminga dwóch 64-bitowych skrótów hex (nieprawidłowe dane → 64)."""
    try:
        return bin((int(str(a), 16) ^ int(str(b), 16)) & 0xFFFFFFFFFFFFFFFF).count("1")
    except (TypeError, ValueError):
        return 64


# ---------------------------------------------------------------- parsowanie

_WS_RE = re.compile(r"\s+")
_SAVED_FROM_RE = re.compile(r"saved\s+from\s+url=\(\d+\)\s*(\S+?)\s*-->", re.I)
_SINGLEFILE_RE = re.compile(r"Page\s+saved\s+with\s+SingleFile\s*(?:\r?\n\s*)?url:\s*(\S+)", re.I)
_CONTACT_SCHEMES = ("tel:", "mailto:", "sms:", "viber:", "whatsapp:", "tg:", "skype:", "signal:")
_CONTACT_HOSTS = frozenset(
    {
        "wa.me",
        "api.whatsapp.com",
        "chat.whatsapp.com",
        "web.whatsapp.com",
        "t.me",
        "telegram.me",
        "signal.me",
        "m.me",
        "invite.viber.com",
    }
)
_JS_POST_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"""(?:\$|jQuery)\.post\(\s*(['"])([^'"\s]{1,500})\1"""),
    re.compile(r"""axios\.post\(\s*(['"])([^'"\s]{1,500})\1"""),
    re.compile(r"""\.open\(\s*['"]POST['"]\s*,\s*(['"])([^'"\s]{1,500})\1""", re.I),
    re.compile(r"""fetch\(\s*(['"])([^'"\s]{1,500})\1\s*,\s*\{[^{}]{0,400}?method\s*:\s*['"]POST['"]""", re.I),
    re.compile(
        r"""\.ajax\(\s*\{(?=[^{}]{0,600}?(?:type|method)\s*:\s*['"]POST['"])[^{}]{0,600}?\burl\s*:\s*(['"])([^'"\s]{1,500})\1""",
        re.I,
    ),
)


def _make_soup(html: str) -> BeautifulSoup | None:
    for parser in ("lxml", "html.parser"):
        try:
            return BeautifulSoup(html, parser)
        except Exception as e:  # noqa: BLE001
            log.debug("BeautifulSoup(%s): %s", parser, e)
    return None


def _clean_text(s: Any, limit: int = 500) -> str | None:
    if s is None:
        return None
    t = _WS_RE.sub(" ", str(s)).strip()
    return t[:limit] if t else None


def _attr(tag: Any, name: str) -> str | None:
    v = tag.get(name) if isinstance(tag, Tag) else None
    if isinstance(v, list):
        v = " ".join(v)
    return v.strip() if isinstance(v, str) and v.strip() else None


def page_url_hint(html: str) -> str | None:
    """Adres oryginalnej strony zapisany w pliku przez przeglądarkę („saved from url”) lub SingleFile."""
    head = html[:5000] if html else ""
    for rx in (_SAVED_FROM_RE, _SINGLEFILE_RE):
        m = rx.search(head)
        if m:
            url = _abs_url(m.group(1), None)
            if url:
                return url
    return None


def empty_parse() -> dict[str, Any]:
    return {
        "title": None,
        "meta_description": None,
        "text": "",
        "tracker_ids": [],
        "script_domains": [],
        "form_actions": [],
        "form_fields": [],
        "outlinks": [],
        "favicon_url": None,
        "dom_simhash": None,
        "html_sha256": None,
        # dodatkowe (poza kontraktem, przydatne w infrastrukturze i UI)
        "lang": None,
        "generator": None,
        "canonical": None,
        "og_url": None,
        "page_url_hint": None,
        "contact_links": [],
        "client_redirect": None,
        "generic": None,
        "element_count": 0,
    }


def parse_html(html: str | bytes | None, base_url: str | None = None) -> dict[str, Any]:
    """Cechy strony z kodu HTML. ``base_url`` — adres strony (do adresów względnych i własnej domeny).
    Nigdy nie rzuca wyjątku; brakujące cechy mają wartości puste."""
    res = empty_parse()
    if html is None:
        return res
    if isinstance(html, (bytes, bytearray)):
        raw_bytes = bytes(html)
        html = decode_html_bytes(raw_bytes)
    else:
        html = str(html)
        raw_bytes = html.encode("utf-8", "replace")
    res["html_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    if not html.strip():
        return res

    try:
        res["page_url_hint"] = page_url_hint(html)
    except Exception as e:  # noqa: BLE001
        log.debug("page_url_hint: %s", e)

    soup = _make_soup(html)
    code_parts: list[str] = []
    if soup is None:
        code_parts.append(html)
        base = base_url or res["page_url_hint"]
    else:
        base = _parse_soup(soup, res, base_url, code_parts)

    code = "\n".join(code_parts)
    try:
        res["tracker_ids"] = unique(extract_tracker_ids(code) + res["tracker_ids"] + click_tracker_ids(code))[:LIST_CAP]
    except Exception as e:  # noqa: BLE001
        log.debug("trackery: %s", e)
    try:
        res["form_actions"] = unique(res["form_actions"] + _js_post_targets(code, base))[:LIST_CAP]
    except Exception as e:  # noqa: BLE001
        log.debug("formularze JS: %s", e)
    try:
        res["client_redirect"] = client_redirect(html, base)
    except Exception as e:  # noqa: BLE001
        log.debug("client_redirect: %s", e)
    if soup is None:
        m = re.search(r"<title[^>]*>(.*?)</title", html, re.I | re.S)
        res["title"] = _clean_text(m.group(1), 300) if m else None
    try:
        res["generic"] = _detect_generic(res["title"], res["text"], html[:50_000])
    except Exception as e:  # noqa: BLE001
        log.debug("generic: %s", e)
    return res


def _js_post_targets(code: str, base: str | None) -> list[str]:
    out: list[str] = []
    for rx in _JS_POST_RES:
        for m in rx.finditer(code):
            url = _abs_url(_js_unescape(m.group(2)), base)
            n = normalize_url(url) if url else None
            if n:
                out.append(n)
    return out


def _parse_soup(soup: BeautifulSoup, res: dict[str, Any], base_url: str | None, code_parts: list[str]) -> str | None:
    """Wypełnia ``res`` cechami z drzewa; zwraca efektywny adres bazowy. Każdy krok osobno — błąd jednego nie
    psuje pozostałych."""
    # <base href> i adres zapisany przez przeglądarkę
    base = base_url or res.get("page_url_hint")
    try:
        bt = soup.find("base", href=True)
        bhref = _attr(bt, "href") if bt else None
        if bhref:
            base = _abs_url(bhref, base) or base
    except Exception as e:  # noqa: BLE001
        log.debug("base: %s", e)
    own = registered_domain(base_url or res.get("page_url_hint") or base)

    # meta: opis, generator, open graph, weryfikacje domen
    metas: dict[str, str] = {}
    try:
        for m in soup.find_all("meta"):
            key = (_attr(m, "name") or _attr(m, "property") or _attr(m, "itemprop") or "").lower()
            content = _attr(m, "content")
            if key and content and key not in metas:
                metas[key] = content
        res["meta_description"] = _clean_text(metas.get("description") or metas.get("og:description"), 500)
        res["generator"] = _clean_text(metas.get("generator"), 120)
        res["og_url"] = _abs_url(metas.get("og:url"), base)
        res["tracker_ids"] += [
            f"{prefix}:{metas[name].strip()}" for name, prefix in _VERIFY_META.items() if metas.get(name)
        ]
    except Exception as e:  # noqa: BLE001
        log.debug("meta: %s", e)

    try:
        t = None
        if soup.head is not None and soup.head.title is not None:
            t = soup.head.title
        if t is None:
            t = soup.find("title")
        title = _clean_text(t.get_text(" ") if t is not None else None, 300)
        res["title"] = title or _clean_text(metas.get("og:title"), 300)
    except Exception as e:  # noqa: BLE001
        log.debug("title: %s", e)

    try:
        html_tag = soup.find("html")
        res["lang"] = _clean_text(_attr(html_tag, "lang"), 20) if html_tag else None
        can = soup.find("link", rel=lambda r: r and "canonical" in (r if isinstance(r, list) else str(r).split()))
        res["canonical"] = _abs_url(_attr(can, "href"), base) if can else None
    except Exception as e:  # noqa: BLE001
        log.debug("lang/canonical: %s", e)

    # kod: treść skryptów + wartości atrybutów (src, href, data-*, content…) — tu szukamy identyfikatorów trackerów
    try:
        for s in soup.find_all("script"):
            if s.string:
                code_parts.append(str(s.string))
        for tag in soup.find_all(True):
            for v in tag.attrs.values():
                if isinstance(v, list):
                    v = " ".join(str(x) for x in v)
                if isinstance(v, str) and v:
                    code_parts.append(v)
    except Exception as e:  # noqa: BLE001
        log.debug("kod: %s", e)

    try:
        doms = []
        for s in soup.find_all("script", src=True):
            d = registered_domain(_abs_url(_attr(s, "src"), base))
            if d and d != own:
                doms.append(d)
        res["script_domains"] = unique(doms)[:LIST_CAP]
    except Exception as e:  # noqa: BLE001
        log.debug("script_domains: %s", e)

    # formularze: adresy wysyłki, nazwy pól, ukryte identyfikatory oferty/strumienia
    try:
        actions, contacts = [], []
        for form in soup.find_all("form"):
            action = _attr(form, "action")
            if not action or action.startswith("#") or action.lower().startswith("javascript:"):
                continue
            if action.lower().startswith("mailto:"):
                contacts.append(unquote(action))
                continue
            n = normalize_url(_abs_url(action, base))
            if n:
                actions.append(n)
        res["form_actions"] = unique(actions)
        names = []
        for el in soup.find_all(["input", "select", "textarea"]):
            name = _attr(el, "name") or _attr(el, "id")
            if name:
                names.append(name.lower()[:64])
            itype = (_attr(el, "type") or "").lower()
            value = _attr(el, "value")
            if (
                el.name == "input"
                and itype == "hidden"
                and name
                and value
                and _CAMPAIGN_FIELD_RE.search(name)
                and not name.lower().startswith("utm_")
                and 3 <= len(value) <= 64
                and not _MACRO_RE.match(value)
                and value.lower() not in ("true", "false", "null", "none", "undefined")
            ):
                res["tracker_ids"].append(f"form:{name.lower()}={value}")
        res["form_fields"] = sorted(set(names))[:LIST_CAP]
        res["contact_links"] = contacts
    except Exception as e:  # noqa: BLE001
        log.debug("formularze: %s", e)

    # linki: domeny wychodzące + linki kontaktowe (tel:, mailto:, wa.me, t.me…)
    try:
        out, contacts = [], []
        for a in soup.find_all("a", href=True):
            href = _attr(a, "href")
            if not href:
                continue
            low = href.lower()
            if low.startswith(_CONTACT_SCHEMES):
                contacts.append(unquote(href)[:300])
                continue
            url = _abs_url(href, base)
            if not url:
                continue
            h = host_of(url)
            if h in _CONTACT_HOSTS:
                contacts.append(url[:300])
            d = registered_domain(h)
            if d and d != own:
                out.append(d)
        res["outlinks"] = unique(out)[:LIST_CAP]
        res["contact_links"] = unique(res["contact_links"] + contacts)[:50]
    except Exception as e:  # noqa: BLE001
        log.debug("linki: %s", e)

    # favicon: <link rel="icon">, potem inne *icon (apple-touch-icon…), potem /favicon.ico
    try:
        best: tuple[int, str] | None = None
        for link in soup.find_all("link", href=True):
            rel = link.get("rel")
            rels = [r.lower() for r in (rel if isinstance(rel, list) else str(rel or "").split())]
            href = _attr(link, "href")
            if not href:
                continue
            prio = 0 if "icon" in rels else (1 if any("icon" in r for r in rels) else None)
            if prio is None:
                continue
            url = href if href.lower().startswith("data:image") else _abs_url(href, base)
            if url and (best is None or prio < best[0]):
                best = (prio, url)
        if best is not None:
            res["favicon_url"] = best[1]
        elif base and base.lower().startswith(("http://", "https://")):
            res["favicon_url"] = urljoin(base, "/favicon.ico")
    except Exception as e:  # noqa: BLE001
        log.debug("favicon: %s", e)

    # szablon DOM
    try:
        paths = _dom_paths(soup)
        res["element_count"] = len(paths)
        res["dom_simhash"] = dom_simhash_from_paths(paths)
    except Exception as e:  # noqa: BLE001
        log.debug("dom_simhash: %s", e)

    # widoczny tekst — na końcu, bo usuwa elementy z drzewa
    try:
        for t in soup.find_all(["script", "style", "noscript", "template", "svg", "head", "iframe", "object", "embed"]):
            t.decompose()
        text = _WS_RE.sub(" ", soup.get_text(" ", strip=True)).strip()
        res["text"] = text[:TEXT_LIMIT]
    except Exception as e:  # noqa: BLE001
        log.debug("tekst: %s", e)
    return base
