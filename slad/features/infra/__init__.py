"""Cechy infrastruktury sieciowej materiału — najsilniej łącząca kampanie grupa cech.

``infra_features(material, ctx) -> (InfraFeatures, page_text)``; kroki (każdy osobno — błąd → wpis w ``errors``):

a) ``meta["infra"]`` — gotowe dane ze zbioru lub generatora syntetycznego (nazwy pól InfraFeatures + synonimy),
b) zapisany HTML (``material.html_path``, także MHTML) — analiza offline,
c) sieć, gdy ``settings.network.enabled`` (w trybie offline — tylko pamięć podręczna, także przeterminowana):
   łańcuch przekierowań i strona docelowa, favicon, DNS, ASN, RDAP, TLS, crt.sh; wyniki w ``ctx.cache(...)``,
d) zawsze: adresy, hosty i domeny z wejścia, łańcucha przekierowań i formularzy.

Ochrona przed fałszywymi powiązaniami: strony platform (Facebook, TikTok, Telegram…) nie są pobierane; strony
„ogólne” (parking domeny, zawieszone konto, błąd HTTP, ochrona przed botami) nie dają cech treści (szablon,
favicon, trackery należą wtedy do hostingu, nie do kampanii); domeny specjalne (.example, .test…) nie są odpytywane.

Drugi element wyniku to widoczny tekst strony docelowej (+ linki kontaktowe tel:/mailto:/wa.me/t.me z kodu
strony, żeby moduł tekstu znalazł w nich IOC).
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slad.models import InfraFeatures, Material

if TYPE_CHECKING:  # pragma: no cover
    from slad.features import ExtractContext

log = logging.getLogger(__name__)

__all__ = ["infra_features"]

FETCH_CACHE_VERSION = 1
NEGATIVE_TTL_H = 2.0  # nieudane zapytanie sieciowe ponawiamy po tylu godzinach (nie po pełnym TTL)
MAX_IPS_FOR_ASN = 3
MAX_RDAP_DOMAINS = 3
LOCAL_HTML_MAX_BYTES = 10_000_000
PAGE_TEXT_MAX = 24_000
LIST_CAP = 200

# ---------------------------------------------------------------- pomocnicze: wartości


def _now() -> datetime:
    return datetime.now(UTC)


_DT_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y.%m.%d %H:%M:%S",
    "%Y.%m.%d",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%Y%m%d",
    "%d-%b-%Y",
    "%d %b %Y",
    "%b %d %Y",
)


def parse_datetime(value: Any) -> datetime | None:
    """Data w dowolnym popularnym formacie (ISO, „01.10.2025”, „2025.10.01 12:00:00”, znacznik czasu…) →
    ``datetime`` ze strefą UTC. Zawsze ze strefą — porównania dat różnych materiałów nie mieszają stref."""
    if value is None or isinstance(value, bool):
        return None
    dt: datetime | None = None
    try:
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, date):
            dt = datetime(value.year, value.month, value.day)
        elif isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(value / 1000 if value > 1e11 else value, UTC)
        else:
            s = str(value).strip()
            if not s:
                return None
            if s.isdigit() and len(s) in (10, 13):
                n = int(s)
                dt = datetime.fromtimestamp(n / 1000 if n > 1e11 else n, UTC)
            else:
                try:
                    dt = datetime.fromisoformat(s[:-1] + "+00:00" if s.endswith(("Z", "z")) else s)
                except ValueError:
                    for fmt in _DT_FORMATS:
                        try:
                            dt = datetime.strptime(s, fmt)
                            break
                        except ValueError:
                            continue
    except (OverflowError, OSError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC)
    return dt if 1985 <= dt.year <= 2100 else None


_URL_SEP = re.compile(r"\s*(?:->|→|=>|\||;|,|\s)\s*")
_TOKEN_SEP = re.compile(r"[\s,;|]+")
_LINE_SEP = re.compile(r"\s*[,;|\n]\s*")
_ASN_SEP = re.compile(r"\s*[,;\n]\s*")  # „13335 | CLOUDFLARENET” to jedna wartość
_URLISH_RE = re.compile(
    r"^(?:(?:https?|hxxps?)://)?(?:(?:[\w\-]+(?:\.|\[\.\]))+[^\W\d_]{2,}|\d{1,3}(?:\.\d{1,3}){3})(?::\d+)?(?:[/?#]\S*)?$",
    re.I | re.UNICODE,
)


def _as_list(value: Any, sep: re.Pattern[str] = _TOKEN_SEP) -> list[str]:
    """Lista tekstów z listy, krotki, zbioru, tekstu JSON lub tekstu rozdzielanego separatorami."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        out: list[str] = []
        for v in value:
            out += _as_list(v, sep)
        return out
    if isinstance(value, dict):
        return []
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            return _as_list(json.loads(s), sep)
        except ValueError:
            pass
    return [t for t in sep.split(s) if t and t.strip()]


def _norm_asn(value: str) -> str | None:
    m = re.match(r"(?i)^\s*(?:AS)?\s*(\d{1,10})\b[\s|:-]*(.*)$", value)
    if not m:
        return value.strip() or None
    name = m.group(2).strip()
    return f"AS{int(m.group(1))}" + (f" {name}" if name else "")


def _norm_ip(value: str) -> str | None:
    try:
        return str(ipaddress.ip_address(value.strip().strip("[]")))
    except ValueError:
        return None


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


# ---------------------------------------------------------------- budowanie wyniku


class _Builder:
    """Scalanie danych z wielu źródeł: listy — bez powtórzeń, w kolejności; pola pojedyncze — pierwsze źródło
    wygrywa (kolejność wywołań = priorytet: meta.infra > zapisany HTML > sieć)."""

    def __init__(self) -> None:
        self.inf = InfraFeatures()
        self.cache_hits = 0
        self.live_calls = 0
        self.timestamps: list[datetime] = []

    def add(self, field: str, values: Iterable[Any]) -> None:
        cur: list[Any] = getattr(self.inf, field)
        seen = set(cur)
        for v in values:
            if v is None or v == "" or v in seen:
                continue
            if len(cur) >= LIST_CAP:
                break
            seen.add(v)
            cur.append(v)

    def set_if_empty(self, field: str, value: Any) -> None:
        if value is None or value == "":
            return
        if getattr(self.inf, field) in (None, ""):
            setattr(self.inf, field, value)

    def error(self, msg: str) -> None:
        if msg and msg not in self.inf.errors:
            self.inf.errors.append(msg)

    def stamp(self, value: Any) -> None:
        dt = parse_datetime(value)
        if dt:
            self.timestamps.append(dt)


_SEED_ALIASES: dict[str, tuple[str, ...]] = {
    "input_urls": ("input_urls", "urls", "url", "ad_url", "link"),
    "redirect_chain": ("redirect_chain", "redirects", "chain", "redirect_urls", "przekierowania"),
    "final_url": ("final_url", "landing_url", "landing", "final", "target_url", "landing_page"),
    "http_status": ("http_status", "status_code", "status"),
    "hosts": ("hosts", "host", "hostname"),
    "domains": ("domains", "domain", "domena", "domeny"),
    "title": ("title", "page_title", "tytul", "tytuł"),
    "meta_description": ("meta_description", "description", "opis"),
    "html_sha256": ("html_sha256",),
    "dom_simhash": ("dom_simhash", "simhash"),
    "tracker_ids": ("tracker_ids", "trackers", "analytics_ids", "tracking_ids"),
    "script_domains": ("script_domains",),
    "form_actions": ("form_actions", "form_action"),
    "form_fields": ("form_fields",),
    "outlinks": ("outlinks",),
    "favicon_mmh3": ("favicon_mmh3", "favicon_hash", "favicon"),
    "ips": ("ips", "ip", "ip_address", "ip_addresses", "adres_ip"),
    "asns": ("asns", "asn", "as"),
    "nameservers": ("nameservers", "ns", "name_servers"),
    "mx": ("mx",),
    "registrar": ("registrar", "rejestrator"),
    "domain_created": (
        "domain_created",
        "created",
        "creation_date",
        "registration_date",
        "registered",
        "data_rejestracji",
    ),
    "tls_issuer": ("tls_issuer", "issuer", "cert_issuer", "ssl_issuer"),
    "tls_fingerprint": ("tls_fingerprint", "cert_sha256", "cert_fingerprint", "fingerprint"),
    "tls_sans": ("tls_sans", "sans", "cert_sans"),
    "ct_domains": ("ct_domains",),
    "fetched_at": ("fetched_at",),
}
_SEED_TRACKER_KEYS: dict[str, str] = {
    "ga": "ga",
    "ga_id": "ga",
    "gtm": "gtm",
    "gtm_id": "gtm",
    "gtag": "gtag",
    "fb_pixel": "fbpixel",
    "fbpixel": "fbpixel",
    "pixel_id": "fbpixel",
    "facebook_pixel": "fbpixel",
    "ym": "ym",
    "yandex_metrika": "ym",
    "tiktok_pixel": "tiktok",
}
_SEED_TEXT_KEYS = ("page_text", "visible_text", "html_text", "text")
# Kolumny rekordu (poza meta["infra"]) jednoznacznie opisujące infrastrukturę — też je wczytujemy. Nazwy ogólne
# („title”, „description”, „status”, „text”) pomijamy: w rekordzie reklamy znaczą co innego (tytuł reklamy itd.).
_TOP_LEVEL_INFRA_KEYS = frozenset(
    {
        "redirect_chain",
        "redirects",
        "redirect_urls",
        "przekierowania",
        "final_url",
        "landing_url",
        "landing_page",
        "ip",
        "ips",
        "ip_address",
        "ip_addresses",
        "adres_ip",
        "asn",
        "asns",
        "nameservers",
        "name_servers",
        "mx",
        "registrar",
        "rejestrator",
        "domain_created",
        "creation_date",
        "registration_date",
        "data_rejestracji",
        "tls_issuer",
        "cert_issuer",
        "ssl_issuer",
        "tls_fingerprint",
        "cert_sha256",
        "cert_fingerprint",
        "tls_sans",
        "favicon_mmh3",
        "favicon_hash",
        "tracker_ids",
        "trackers",
        "tracking_ids",
        "analytics_ids",
        "fb_pixel",
        "fbpixel",
        "pixel_id",
        "facebook_pixel",
        "ga_id",
        "gtm_id",
        "tiktok_pixel",
        "yandex_metrika",
        "dom_simhash",
        "html_sha256",
        "page_title",
        "http_status",
        "status_code",
    }
)


def _apply_seed_field(b: _Builder, field: str, value: Any) -> None:
    from slad.features.infra.domains import clean_host, host_of, normalize_url, registered_domain
    from slad.features.infra.favicon import favicon_hash_from_data_uri
    from slad.features.infra.html import normalize_tracker_id

    if field in ("input_urls", "redirect_chain", "form_actions"):
        b.add(field, (normalize_url(u) for u in _as_list(value, _URL_SEP)))
    elif field == "final_url":
        urls = [normalize_url(u) for u in _as_list(value, _URL_SEP)]
        b.set_if_empty(field, next((u for u in urls if u), None))
    elif field == "hosts":
        b.add(field, (host_of(x) for x in _as_list(value)))
    elif field in ("domains", "script_domains", "outlinks"):
        b.add(field, (registered_domain(x) or clean_host(x) for x in _as_list(value)))
    elif field == "mx":
        b.add(field, (clean_host(x.split()[-1]) for x in _as_list(value, _LINE_SEP) if x.split()))
    elif field in ("nameservers", "tls_sans", "ct_domains"):
        b.add(field, (clean_host(x.lstrip("*.")) for x in _as_list(value)))
    elif field == "ips":
        b.add(field, (_norm_ip(x) for x in _as_list(value)))
    elif field == "asns":
        b.add(field, (_norm_asn(x) for x in _as_list(value, _ASN_SEP)))
    elif field == "tracker_ids":
        b.add(field, (normalize_tracker_id(x) for x in _as_list(value)))
    elif field == "form_fields":
        b.add(field, sorted({x.strip().lower() for x in _as_list(value) if x.strip()}))
    elif field == "favicon_mmh3":
        s = str(value).strip()
        if re.fullmatch(r"[-+]?\d{1,11}", s):
            b.set_if_empty(field, int(s))
        elif s.lower().startswith("data:"):
            b.set_if_empty(field, favicon_hash_from_data_uri(s))
    elif field == "http_status":
        s = str(value).strip()
        if re.fullmatch(r"\d{3}", s):
            b.set_if_empty(field, int(s))
    elif field in ("domain_created", "fetched_at"):
        b.set_if_empty(field, parse_datetime(value))
    elif field == "tls_fingerprint":
        fp = re.sub(r"[^0-9a-f]", "", str(value).lower())
        b.set_if_empty(field, fp or None)
    else:  # title, meta_description, html_sha256, dom_simhash, registrar, tls_issuer
        s = re.sub(r"\s+", " ", str(value)).strip()
        b.set_if_empty(field, s[:500] or None)


def _apply_seed(b: _Builder, seed: Any, texts: list[str], fetch_candidates: list[str]) -> str | None:
    """meta["infra"] → pola InfraFeatures. Zwraca surowy HTML, jeśli zbiór go zawiera (analizowany jak plik)."""
    from slad.features.infra.domains import prepare_fetch_url
    from slad.features.infra.html import normalize_tracker_id

    if isinstance(seed, str):
        try:
            seed = json.loads(seed)
        except ValueError:
            b.error("meta.infra: nieprawidłowy format (oczekiwano JSON)")
            return None
    if not isinstance(seed, dict):
        b.error("meta.infra: nieprawidłowy format (oczekiwano słownika)")
        return None
    lower = {str(k).strip().lower(): v for k, v in seed.items()}
    for field, aliases in _SEED_ALIASES.items():
        for alias in aliases:
            value = lower.get(alias)
            if value is None or value == "" or value == [] or value == {}:
                continue
            try:
                _apply_seed_field(b, field, value)
            except Exception as e:  # noqa: BLE001
                b.error(f"meta.infra.{alias}: nieprawidłowa wartość ({type(e).__name__})")
            if field == "input_urls":
                fetch_candidates += [u for u in (prepare_fetch_url(x) for x in _as_list(value, _URL_SEP)) if u]
            break
    for key, kind in _SEED_TRACKER_KEYS.items():
        if key in lower:
            b.add("tracker_ids", (normalize_tracker_id(v, kind) for v in _as_list(lower[key])))
    for key in ("errors", "error"):  # błędy z chwili zbierania danych (np. martwa strona w zbiorze)
        for msg in _as_list(lower.get(key), _LINE_SEP if key == "error" else re.compile(r"\n")):
            b.error(msg.strip()[:300])
    for key in _SEED_TEXT_KEYS:
        v = lower.get(key)
        if isinstance(v, str) and v.strip():
            texts.append(v.strip())
            break
    html = lower.get("html")
    return html if isinstance(html, str) and "<" in html else None


# ---------------------------------------------------------------- adresy wejściowe

_URL_KEY_HINTS = (
    "url",
    "link",
    "domain",
    "domena",
    "adres",
    "strona",
    "landing",
    "redirect",
    "przekier",
    "website",
    "href",
)
_URL_KEY_EXCLUDE = (
    "image",
    "img",
    "obraz",
    "zdj",
    "photo",
    "screenshot",
    "zrzut",
    "thumb",
    "file",
    "plik",
    "path",
    "ścież",
    "sciez",
    "video",
    "wideo",
    "avatar",
    "logo",
    "icon",
    "media",
)


def _input_candidates(material: Material) -> list[str]:
    """Surowe adresy z wejścia: Material.url (może zawierać kilka) + pola meta wyglądające na adres/domenę."""
    from slad.features.infra.domains import unique

    cands: list[str] = []
    if material.url:
        cands += [u for u in _as_list(material.url, _URL_SEP) if _URLISH_RE.match(u)]
    meta = material.meta if isinstance(material.meta, dict) else {}
    for k, v in meta.items():
        lk = str(k).lower()
        if lk == "infra" or not any(h in lk for h in _URL_KEY_HINTS) or any(x in lk for x in _URL_KEY_EXCLUDE):
            continue
        if isinstance(v, (str, list, tuple)):
            cands += [u for u in _as_list(v, _URL_SEP) if len(u) < 2048 and _URLISH_RE.match(u)]
    return unique(cands)


# ---------------------------------------------------------------- pamięć podręczna


def _cache(ctx: Any, name: str) -> Any:
    fn = getattr(ctx, "cache", None)
    if fn is None:
        return None
    try:
        return fn(name)
    except Exception as e:  # noqa: BLE001
        log.debug("cache %s niedostępny: %s", name, e)
        return None


_DATA_KEYS = ("status", "a", "aaaa", "ns", "mx", "fingerprint", "registrar", "created", "asn", "hash", "domains")


def _is_failure(value: dict[str, Any]) -> bool:
    if not (value.get("error") or value.get("errors")):
        return False
    return not any(value.get(k) for k in _DATA_KEYS)


def _cached(
    b: _Builder,
    ctx: Any,
    name: str,
    key: str,
    online: bool,
    compute: Callable[[], dict[str, Any] | None],
    valid: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any] | None:
    """Wartość z pamięci podręcznej albo (gdy online) świeżo policzona i zapisana. Offline — także przeterminowana.
    Nieudane zapytania (sam błąd, bez danych) są ponawiane po NEGATIVE_TTL_H godzinach; wpisy, których ``valid``
    nie akceptuje (np. stary format), są pomijane."""
    cache = _cache(ctx, name)
    if cache is not None:
        hit = cache.get(key)
        if isinstance(hit, dict) and (valid is None or valid(hit)):
            stale_failure = False
            if online and _is_failure(hit):
                age = cache.age_h(key) if hasattr(cache, "age_h") else None
                stale_failure = age is not None and age > NEGATIVE_TTL_H
            if not stale_failure:
                b.cache_hits += 1
                b.stamp(hit.get("fetched_at"))
                return hit
    if not online:
        if cache is not None:
            try:
                stale = cache.get(key, allow_expired=True)
            except TypeError:  # obca implementacja cache bez allow_expired
                stale = None
            if isinstance(stale, dict) and (valid is None or valid(stale)):
                b.cache_hits += 1
                b.stamp(stale.get("fetched_at"))
                return stale
        return None
    value = compute()
    if value is None:
        return None
    value.setdefault("fetched_at", _iso(_now()))
    b.live_calls += 1
    b.stamp(value.get("fetched_at"))
    if cache is not None:
        try:
            cache.set(key, value)
        except Exception as e:  # noqa: BLE001
            log.debug("cache %s: zapis nieudany: %s", name, e)
    return value


# ---------------------------------------------------------------- strony (lokalne i pobrane)

_PAGE_LIST_FIELDS = ("tracker_ids", "script_domains", "form_actions", "form_fields", "outlinks")
_PAGE_SCALAR_FIELDS = ("title", "meta_description", "html_sha256", "dom_simhash")


def _apply_page(b: _Builder, parsed: dict[str, Any], *, scalars: bool) -> None:
    for f in _PAGE_LIST_FIELDS:
        b.add(f, parsed.get(f) or [])
    if scalars:
        for f in _PAGE_SCALAR_FIELDS:
            b.set_if_empty(f, parsed.get(f))


def _page_texts(parsed: dict[str, Any]) -> list[str]:
    out = []
    if parsed.get("text"):
        out.append(parsed["text"])
    if parsed.get("contact_links"):
        out.append(" ".join(parsed["contact_links"]))
    return out


def _read_local_html(path_str: str, ctx: Any) -> tuple[str, str | None]:
    """Zapisany plik strony (HTML lub MHTML) → (HTML, adres oryginalnej strony z pliku | None)."""
    from slad.features.infra.html import decode_html_bytes

    p = Path(path_str)
    if not p.is_absolute() and not p.exists():
        settings = getattr(ctx, "settings", None)
        if settings is not None and hasattr(settings, "path"):
            p = settings.path(p)
    with open(p, "rb") as fh:
        data = fh.read(LOCAL_HTML_MAX_BYTES)
    head = data[:8192].lower()
    if b"mime-version:" in head and b"multipart/related" in head:
        html, loc = _mhtml_html(data)
        if html is not None:
            return html, loc
    return decode_html_bytes(data), None


def _mhtml_html(data: bytes) -> tuple[str | None, str | None]:
    import email
    from email import policy

    from slad.features.infra.html import decode_html_bytes

    try:
        msg = email.message_from_bytes(data, policy=policy.compat32)
        loc = msg.get("Snapshot-Content-Location")
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset()
                html = decode_html_bytes(payload, f"text/html; charset={charset}" if charset else None)
                return html, str(loc or part.get("Content-Location") or "") or None
    except Exception as e:  # noqa: BLE001
        log.debug("MHTML: %s", e)
    return None, None


def _fetch_entry(res: Any) -> dict[str, Any]:
    """FetchResult → wpis pamięci podręcznej: przeanalizowane strony zamiast surowego HTML."""
    from slad.features.infra.html import generic_page_reason, parse_html

    pages = []
    for url, body in res.pages:
        parsed = parse_html(body, base_url=url)
        parsed["url"] = url
        pages.append(parsed)
    final = pages[-1] if pages and pages[-1].get("url") == res.final_url else None
    generic = generic_page_reason(final, res.status) if res.status is not None else None
    return {
        "v": FETCH_CACHE_VERSION,
        "chain": list(res.chain),
        "final_url": res.final_url,
        "status": res.status,
        "content_type": res.content_type,
        "error": res.error,
        "tls_invalid": bool(res.tls_invalid),
        "truncated": bool(getattr(res, "truncated", False)),
        "server": (res.headers or {}).get("server"),
        "powered_by": (res.headers or {}).get("x-powered-by"),
        "pages": pages,
        "generic": generic,
        "fetched_at": _iso(_now()),
    }


def _select_target(candidates: list[str]) -> tuple[str | None, list[str], bool]:
    """Pierwszy adres do pobrania: opakowania przekierowań platform są rozpakowywane (opakowanie trafia do
    łańcucha), strony platform pomijane. Zwraca (adres, prefiks łańcucha, czy pominięto adres platformy)."""
    from slad.features.infra.domains import is_platform_url, normalize_url, unwrap_redirector

    skipped = False
    for u in candidates:
        if not u:
            continue
        inner = unwrap_redirector(u)
        cand, pre = (inner, [normalize_url(u) or u]) if inner else (u, [])
        if is_platform_url(cand):
            skipped = True
            continue
        return cand, pre, skipped
    return None, [], skipped


# ---------------------------------------------------------------- główna funkcja


def infra_features(material: Material, ctx: ExtractContext) -> tuple[InfraFeatures, str]:
    """Cechy infrastruktury materiału + widoczny tekst strony docelowej. Nigdy nie rzuca wyjątku z powodu
    pojedynczego kroku — brakujące dane zostają puste, a przyczyna trafia do ``errors``."""
    from slad.features.infra.domains import unique

    b = _Builder()
    texts: list[str] = []
    try:
        _extract(material, ctx, b, texts)
    except Exception as e:  # noqa: BLE001 — ostatnia linia obrony: zwracamy to, co udało się zebrać
        log.warning("infra_features(%s): %s", material.id, e, exc_info=True)
        b.error(f"infrastruktura: błąd wewnętrzny ({type(e).__name__})")
    inf = b.inf
    if b.timestamps and inf.fetched_at is None:
        inf.fetched_at = min(b.timestamps)
    inf.from_cache = b.cache_hits > 0 and b.live_calls == 0
    page_text = "\n".join(unique(t.strip() for t in texts if t and t.strip()))[:PAGE_TEXT_MAX]
    return inf, page_text


def _extract(material: Material, ctx: Any, b: _Builder, texts: list[str]) -> None:
    from slad.features.infra.domains import (
        host_of,
        is_platform_url,
        is_reserved_host,
        normalize_url,
        prepare_fetch_url,
        registered_domain,
        unique,
    )
    from slad.features.infra.html import page_url_hint, parse_html, url_tracker_ids

    inf = b.inf
    cfg = getattr(getattr(ctx, "settings", None), "network", None)
    if cfg is None:
        from slad.settings import NetworkSettings

        cfg = NetworkSettings()
    online = bool(getattr(cfg, "enabled", False))
    fetch_candidates: list[str] = []

    # (a) gotowe dane ze zbioru: meta["infra"], potem jednoznaczne kolumny infrastruktury z samego meta
    seed_html: str | None = None
    meta = material.meta if isinstance(material.meta, dict) else {}
    if meta.get("infra") is not None:
        try:
            seed_html = _apply_seed(b, meta["infra"], texts, fetch_candidates)
        except Exception as e:  # noqa: BLE001
            b.error(f"meta.infra: nieprawidłowe dane ({type(e).__name__})")
    top_seed = {k: v for k, v in meta.items() if str(k).strip().lower() in _TOP_LEVEL_INFRA_KEYS}
    if top_seed:
        try:
            _apply_seed(b, top_seed, texts, fetch_candidates)
        except Exception as e:  # noqa: BLE001
            b.error(f"meta: nieprawidłowe dane infrastruktury ({type(e).__name__})")

    # adresy z wejścia (Material.url + pola meta)
    try:
        raw = _input_candidates(material)
        b.add("input_urls", (normalize_url(u) for u in raw))
        fetch_candidates[:0] = [u for u in (prepare_fetch_url(x) for x in raw) if u]
    except Exception as e:  # noqa: BLE001
        b.error(f"adresy wejściowe: {type(e).__name__}")

    # (b) zapisany HTML — offline; adres bazowy = adres strony docelowej (nie opakowanie platformy)
    pre_target = _select_target(fetch_candidates)[0]
    local: dict[str, Any] | None = None
    html_sources: list[tuple[str, str | None, str]] = []
    if seed_html:
        html_sources.append((seed_html, None, "meta.infra.html"))
    if material.html_path:
        try:
            html, loc = _read_local_html(material.html_path, ctx)
            html_sources.append((html, loc, Path(material.html_path).name))
        except FileNotFoundError:
            b.error(f"html: brak pliku {Path(material.html_path).name}")
        except Exception as e:  # noqa: BLE001
            b.error(f"html: nie można odczytać pliku {Path(material.html_path).name} ({type(e).__name__})")
    for html, loc, label in html_sources:
        try:
            hint = loc or page_url_hint(html)
            base = hint or inf.final_url or (inf.redirect_chain[-1] if inf.redirect_chain else None) or pre_target
            parsed = parse_html(html, base_url=base)
            if hint:
                b.add("input_urls", [normalize_url(hint)])
                fetch_candidates.append(prepare_fetch_url(hint) or "")
            if parsed.get("generic"):
                b.error(f"html ({label}): {parsed['generic']} — pominięto cechy treści strony")
                continue
            _apply_page(b, parsed, scalars=True)
            texts += _page_texts(parsed)
            local = local or parsed
        except Exception as e:  # noqa: BLE001
            b.error(f"html ({label}): błąd analizy ({type(e).__name__})")

    # (c) sieć / pamięć podręczna
    fetch_candidates = unique(u for u in fetch_candidates if u)
    target, prefix, skipped_platform = _select_target(fetch_candidates)
    if target and prefix:  # opakowanie platformy (l.facebook.com/l.php?u=…) zdradza stronę docelową także offline
        b.add("redirect_chain", prefix + [normalize_url(target) or target])
        fetch_candidates.append(target)
    if target is None and skipped_platform:
        b.error("pominięto pobieranie: adres platformy (Facebook, TikTok, Telegram…) — brak strony docelowej")
    entry: dict[str, Any] | None = None
    if target and not is_reserved_host(target):
        try:
            entry = _step_fetch(b, ctx, cfg, online, target, prefix)
        except Exception as e:  # noqa: BLE001
            log.debug("fetch: %s", e, exc_info=True)
            b.error(f"pobieranie: błąd ({type(e).__name__})")
        if entry is not None:
            texts += _apply_fetch_entry(b, entry, has_local=local is not None)
        elif not online:
            b.error("tryb offline: pominięto pobieranie strony i zapytania sieciowe (brak danych w pamięci podręcznej)")

    final_page = None
    if entry and entry.get("pages") and entry["pages"][-1].get("url") == entry.get("final_url"):
        final_page = None if entry.get("generic") else entry["pages"][-1]

    net_host = (host_of(inf.final_url) or host_of(target)) if (inf.final_url or target) else None
    if net_host and not is_reserved_host(net_host) and not is_platform_url(net_host):
        responded = bool(entry and entry.get("status") is not None)
        for step in (_step_favicon, _step_dns, _step_rdap, _step_tls, _step_crtsh):
            try:
                step(b, ctx, cfg, online, net_host=net_host, local=local, final_page=final_page, responded=responded)
            except Exception as e:  # noqa: BLE001
                log.debug("%s: %s", step.__name__, e, exc_info=True)
                b.error(f"{step.__name__.removeprefix('_step_')}: błąd ({type(e).__name__})")
    elif inf.favicon_mmh3 is None:
        try:  # favicona osadzona w HTML (data:) — bez sieci
            _favicon_from_pages(b, [local])
        except Exception as e:  # noqa: BLE001
            b.error(f"favicon: błąd ({type(e).__name__})")

    # (d) adresy, hosty i domeny z wejścia, łańcucha i formularzy
    try:
        urls = list(inf.input_urls) + list(inf.redirect_chain)
        if inf.final_url:
            urls.append(inf.final_url)
        urls += list(inf.form_actions)
        b.add("hosts", (host_of(u) for u in urls))
        b.add("domains", (registered_domain(u) for u in urls))
        b.add("tracker_ids", url_tracker_ids(list(inf.input_urls) + list(inf.redirect_chain) + fetch_candidates))
    except Exception as e:  # noqa: BLE001
        b.error(f"domeny: {type(e).__name__}")


# ---------------------------------------------------------------- kroki sieciowe


def _step_fetch(b: _Builder, ctx: Any, cfg: Any, online: bool, target: str, prefix: list[str]) -> dict[str, Any] | None:
    from slad.features.infra.domains import normalize_url
    from slad.features.infra.fetch import fetch_chain

    key = normalize_url(target) or target

    def compute() -> dict[str, Any]:
        return _fetch_entry(fetch_chain(target, cfg))

    entry = _cached(b, ctx, "fetch", key, online, compute, valid=lambda v: v.get("v") == FETCH_CACHE_VERSION)
    if entry is None:
        return None
    entry = dict(entry)
    entry["prefix"] = prefix
    return entry


def _apply_fetch_entry(b: _Builder, entry: dict[str, Any], *, has_local: bool) -> list[str]:
    """Wpis pobrania → pola InfraFeatures; zwraca teksty strony do page_text."""
    from slad.features.infra.domains import normalize_url

    b.add("redirect_chain", (normalize_url(u) or u for u in list(entry.get("prefix") or []) + entry.get("chain", [])))
    b.set_if_empty("final_url", normalize_url(entry.get("final_url")) or entry.get("final_url"))
    b.set_if_empty("http_status", entry.get("status"))
    if entry.get("error"):
        b.error(f"pobieranie: {entry['error']}")
    if entry.get("tls_invalid"):
        b.error("tls: nieprawidłowy certyfikat (strona pobrana bez weryfikacji)")
    pages: list[dict[str, Any]] = entry.get("pages") or []
    final_url = entry.get("final_url")
    generic = entry.get("generic")
    texts: list[str] = []
    for i, page in enumerate(pages):
        is_final = i == len(pages) - 1 and page.get("url") == final_url
        if is_final:
            if generic:
                continue
            _apply_page(b, page, scalars=True)
            if not has_local:
                texts += _page_texts(page)
        elif not page.get("generic"):
            _apply_page(b, page, scalars=False)  # strony pośrednie (meta refresh / JS): tylko listy
    if generic:
        b.error(f"strona docelowa: {generic} — pominięto cechy treści strony")
    return texts


def _favicon_from_pages(b: _Builder, pages: list[dict[str, Any] | None]) -> bool:
    from slad.features.infra.favicon import favicon_hash_from_data_uri

    for page in pages:
        fav = (page or {}).get("favicon_url")
        if fav and fav.lower().startswith("data:"):
            h = favicon_hash_from_data_uri(fav)
            if h is not None:
                b.set_if_empty("favicon_mmh3", h)
                return True
    return False


def _step_favicon(b: _Builder, ctx: Any, cfg: Any, online: bool, **kw: Any) -> None:
    from urllib.parse import urljoin

    from slad.features.infra.domains import is_reserved_host, normalize_url, unique
    from slad.features.infra.favicon import favicon_hash, is_image
    from slad.features.infra.fetch import fetch_bytes

    if b.inf.favicon_mmh3 is not None:
        return
    pages = [kw.get("local"), kw.get("final_page")]
    if _favicon_from_pages(b, pages):
        return
    if not getattr(cfg, "favicon", True):
        return
    urls: list[str] = []
    for page in pages:
        fav = (page or {}).get("favicon_url")
        if fav and fav.lower().startswith(("http://", "https://")):
            urls.append(fav)
            urls.append(urljoin(fav, "/favicon.ico"))
    for fav in unique(urls):
        key = normalize_url(fav)
        if not key or is_reserved_host(key):
            continue
        can_fetch = online and kw.get("responded", False)

        def compute(url: str = key) -> dict[str, Any]:
            data, ctype, err = fetch_bytes(url, cfg)
            if data is not None and is_image(data, ctype):
                return {"hash": favicon_hash(data), "error": None, "content_type": ctype}
            return {"hash": None, "error": err or "to nie jest obraz", "content_type": ctype}

        value = _cached(b, ctx, "favicon", key, can_fetch, compute)
        if value and value.get("hash") is not None:
            b.set_if_empty("favicon_mmh3", int(value["hash"]))
            return
    if urls and online and kw.get("responded"):
        b.error("favicon: brak lub nieprawidłowy plik")


def _step_dns(b: _Builder, ctx: Any, cfg: Any, online: bool, **kw: Any) -> None:
    from slad.features.infra.netinfo import asn_lookup, dns_info, rdap_ip

    if not getattr(cfg, "dns", True):
        return
    host = kw["net_host"]
    timeout = float(getattr(cfg, "timeout_s", 8.0))
    value = _cached(b, ctx, "dns", host, online, lambda: dns_info(host, timeout))
    if not value:
        return
    b.add("ips", (value.get("a") or []) + (value.get("aaaa") or []))
    b.add("nameservers", value.get("ns") or [])
    b.add("mx", value.get("mx") or [])
    for err in value.get("errors") or []:
        b.error(err)

    ips = list(value.get("a") or [])[:MAX_IPS_FOR_ASN] or list(value.get("aaaa") or [])[:1]
    for ip in ips:

        def compute(ip: str = ip) -> dict[str, Any]:
            asn, err = asn_lookup(ip, timeout)
            if asn is None and err and getattr(cfg, "rdap", True):
                net = rdap_ip(ip, timeout)
                if net.get("name"):
                    asn = f"{net['asn']} {net['name']}" if net.get("asn") else str(net["name"])
                    err = None
            return {"asn": asn, "error": err}

        v = _cached(b, ctx, "asn", ip, online, compute)
        if v and v.get("asn"):
            b.add("asns", [v["asn"]])
        elif v and v.get("error"):
            b.error(v["error"])


def _step_rdap(b: _Builder, ctx: Any, cfg: Any, online: bool, **kw: Any) -> None:
    from slad.features.infra.domains import (
        is_platform_url,
        is_reserved_host,
        is_stoplisted_domain,
        registered_domain,
        unique,
    )
    from slad.features.infra.netinfo import rdap_domain

    if not getattr(cfg, "rdap", True):
        return
    inf = b.inf
    timeout = float(getattr(cfg, "timeout_s", 8.0))
    urls = ([inf.final_url] if inf.final_url else []) + list(reversed(inf.redirect_chain)) + list(inf.input_urls)
    domains = [registered_domain(u) for u in urls] + [registered_domain(kw["net_host"])]
    candidates = [
        d
        for d in unique(domains)
        if d and not is_stoplisted_domain(d) and not is_platform_url(d) and not is_reserved_host(d)
    ][:MAX_RDAP_DOMAINS]
    first_error = None
    for d in candidates:
        value = _cached(b, ctx, "rdap", d, online, lambda d=d: rdap_domain(d, timeout))
        if not value:
            continue
        if value.get("registrar") or value.get("created"):
            b.set_if_empty("registrar", value.get("registrar"))
            b.set_if_empty("domain_created", parse_datetime(value.get("created")))
            b.add("nameservers", value.get("nameservers") or [])
            return
        first_error = first_error or value.get("error")
    if first_error:
        b.error(first_error)


def _step_tls(b: _Builder, ctx: Any, cfg: Any, online: bool, **kw: Any) -> None:
    from slad.features.infra.tls import tls_info

    if not getattr(cfg, "tls", True):
        return
    url = b.inf.final_url or ""
    if not url.lower().startswith("https://") or not kw.get("responded"):
        return
    host = kw["net_host"]
    from slad.features.infra.fetch import blocked_reason

    reason = blocked_reason(f"https://{host}/")  # ochrona przed SSRF także dla sondy TLS
    if reason:
        b.error(reason)
        return
    timeout = float(getattr(cfg, "timeout_s", 8.0))
    value = _cached(b, ctx, "tls", f"{host}:443", online, lambda: tls_info(host, 443, timeout))
    if not value:
        return
    b.set_if_empty("tls_issuer", value.get("issuer"))
    b.set_if_empty("tls_fingerprint", value.get("fingerprint"))
    b.add("tls_sans", value.get("sans") or [])
    if value.get("error") and not value.get("fingerprint"):
        b.error(value["error"])


def _step_crtsh(b: _Builder, ctx: Any, cfg: Any, online: bool, **kw: Any) -> None:
    from slad.features.infra.domains import registered_domain
    from slad.features.infra.tls import crtsh_lookup

    if not getattr(cfg, "crtsh", False):
        return
    host = kw["net_host"]
    domain = registered_domain(host)
    if not domain:
        return
    timeout = float(getattr(cfg, "timeout_s", 8.0)) * 3

    def compute() -> dict[str, Any]:
        names, err = crtsh_lookup(domain, timeout)
        return {"domains": names, "error": err}

    value = _cached(b, ctx, "crtsh", domain, online, compute)
    if not value:
        return
    b.add("ct_domains", (n for n in value.get("domains") or [] if n != host))
    if value.get("error") and not value.get("domains"):
        b.error(value["error"])
