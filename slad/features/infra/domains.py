"""Adresy URL i domeny: host, domena zarejestrowana (eTLD+1), normalizacja adresów, listy wyjątków.

- Lista sufiksów publicznych (PSL) pochodzi z migawki dołączonej do pakietu tldextract — NIGDY nie jest pobierana
  z sieci (``suffix_list_urls=()``, ``cache_dir=None``).
- Uwzględniamy prywatną część PSL (np. ``netlify.app``, ``github.io``, ``blogspot.com``) oraz popularne darmowe
  kreatory stron spoza PSL: ``foo.netlify.app`` i ``bar.netlify.app`` to dwie RÓŻNE „domeny zarejestrowane”
  (różni właściciele). Inaczej wszystkie oszustwa z jednego darmowego hostingu wyglądałyby na jedną kampanię.
- Adresy „rozbrojone” (``hxxps://scam[.]pl``), częste w zgłoszeniach i zbiorach IOC, są przywracane.
"""

from __future__ import annotations

import ipaddress
import logging
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote_plus, urlsplit, urlunsplit

import tldextract

log = logging.getLogger(__name__)

# Darmowe kreatory / hostingi stron spoza prywatnej części PSL: każda subdomena to osobna strona (osobny właściciel).
SHARED_HOSTING_SUFFIXES: tuple[str, ...] = (
    "000webhostapp.com",
    "weebly.com",
    "wordpress.com",
    "tilda.ws",
    "godaddysites.com",
    "mystrikingly.com",
    "strikingly.com",
    "jimdosite.com",
    "jimdofree.com",
    "webnode.page",
    "site123.me",
    "yolasite.com",
    "zyrosite.com",
    "hostingersite.com",
    "wixstudio.io",
    "ucoz.net",
    "ucoz.ru",
    "narod.ru",
    "hpage.com",
    "mozello.com",
    "webador.com",
    "tiiny.site",
    "lovable.app",
    "softr.app",
    "typedream.app",
    "notion.site",
    "super.site",
    "framer.app",
    "framer.website",
    "my.canva.site",
    "wixsite.com",
)

_EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(),
    cache_dir=None,
    include_psl_private_domains=True,
    extra_suffixes=SHARED_HOSTING_SUFFIXES,
)
try:  # rozgrzewka: wczytanie migawki PSL raz, przed użyciem z wielu wątków
    _EXTRACT("example.com")
except Exception as _e:  # noqa: BLE001
    log.warning("tldextract: nie udało się wczytać listy sufiksów: %s", _e)

# Parametry śledzące, usuwane przy normalizacji (identyfikują kliknięcie, nie kampanię).
TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "fbclid",
        "gclid",
        "gclsrc",
        "dclid",
        "gbraid",
        "wbraid",
        "yclid",
        "ysclid",
        "msclkid",
        "ttclid",
        "twclid",
        "igshid",
        "mc_eid",
        "mc_cid",
        "_ga",
        "_gl",
        "_hsenc",
        "_hsmi",
        "mkt_tok",
    }
)

# Strony platform (portale społecznościowe, komunikatory, sklepy z aplikacjami): nie pobieramy ich — szablon,
# favicon i tytuł są wspólne dla milionów stron i dawałyby fałszywe powiązania. Skracacze linków (bit.ly, t.co…)
# NIE są tu wymienione: je trzeba przejść, żeby poznać stronę docelową.
PLATFORM_DOMAINS: frozenset[str] = frozenset(
    {
        "facebook.com",
        "fb.com",
        "fb.me",
        "fb.watch",
        "instagram.com",
        "threads.net",
        "threads.com",
        "messenger.com",
        "m.me",
        "whatsapp.com",
        "wa.me",
        "t.me",
        "telegram.me",
        "telegram.org",
        "tiktok.com",
        "youtube.com",
        "youtu.be",
        "x.com",
        "twitter.com",
        "linkedin.com",
        "pinterest.com",
        "snapchat.com",
        "reddit.com",
        "vk.com",
        "ok.ru",
        "discord.com",
        "discord.gg",
        "apple.com",
        "signal.me",
        "viber.com",
    }
)
PLATFORM_HOSTS: frozenset[str] = frozenset({"play.google.com", "accounts.google.com", "www.google.com", "google.com"})

# Domeny specjalnego przeznaczenia (RFC 2606 / 6761 / 7686) — nigdy nie istnieją w internecie; nie odpytujemy sieci.
RESERVED_TLDS: frozenset[str] = frozenset(
    {"example", "test", "invalid", "localhost", "local", "internal", "onion", "lan"}
)

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_HOST_RE = re.compile(r"^[a-z0-9_](?:[a-z0-9_\-]{0,62})(?:\.[a-z0-9_](?:[a-z0-9_\-]{0,62}))*$")
_DEFANG_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\s*h(?:xx|\*\*)p(s?)(?=(?::|\[:\])//)", re.I), r"http\1"),
    (re.compile(r"\[\s*\.\s*\]|\(\s*\.\s*\)|\{\s*\.\s*\}|\[dot\]|\(dot\)|\{dot\}", re.I), "."),
    (re.compile(r"\[\s*:\s*\]"), ":"),
    (re.compile(r"\[\s*/\s*\]"), "/"),
)
_PATH_SAFE = "/%:@!$&'()*+,;=~-._"
_QUERY_SAFE = "/%:@!$&'()*+,;=~-._?"


# ---------------------------------------------------------------- pomocnicze


def refang(s: str) -> str:
    """``hxxps://scam[.]pl`` → ``https://scam.pl``."""
    for rx, repl in _DEFANG_RULES:
        s = rx.sub(repl, s)
    return s


def is_ip(value: str | None) -> bool:
    if not value:
        return False
    try:
        ipaddress.ip_address(value.strip("[]"))
        return True
    except ValueError:
        return False


def _to_ascii(host: str) -> str | None:
    """IDN → punycode (IDNA 2008/UTS46, awaryjnie IDNA 2003 z biblioteki standardowej)."""
    try:
        import idna

        return idna.encode(host, uts46=True).decode("ascii").lower()
    except Exception:  # noqa: BLE001
        pass
    try:
        return host.encode("idna").decode("ascii").lower()
    except Exception:  # noqa: BLE001
        return None


def clean_host(host: str | None) -> str | None:
    """Host → małe litery, bez kropki końcowej, IDN jako punycode; IP w postaci kanonicznej; śmieci → None."""
    if not host or not isinstance(host, str):
        return None
    host = host.strip().strip("[]").rstrip(".").lower()
    if not host:
        return None
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass
    if not host.isascii():
        host = _to_ascii(host)
        if not host:
            return None
    if len(host) > 253 or not _HOST_RE.match(host):
        return None
    return host


def _prepare(url: str) -> str | None:
    s = refang(url.strip().strip("\"'<>").strip())
    if not s:
        return None
    if s.startswith("//"):
        s = "http:" + s
    elif not _SCHEME_RE.match(s):
        if re.match(r"^(?:mailto|tel|sms|javascript|data|tg|whatsapp|viber|intent|market):", s, re.I):
            return s
        s = "http://" + s
    return s


def _split(url: str):
    s = _prepare(url)
    if s is None:
        return None
    try:
        return urlsplit(s)
    except ValueError:
        return None


# ---------------------------------------------------------------- API


def host_of(url: str | None) -> str | None:
    """Host z adresu URL (lub samego hosta): małe litery, punycode; None gdy to nie jest adres sieciowy."""
    if not url or not isinstance(url, str):
        return None
    parts = _split(url)
    if parts is None or parts.scheme.lower() not in ("http", "https", "ftp", "ws", "wss"):
        return None
    try:
        raw = parts.hostname
    except ValueError:
        raw = None
    if not raw:
        # urlsplit zwraca hostname w małych literach; gdy netloc jest dziwny (np. zły port) — ręcznie
        netloc = parts.netloc.rsplit("@", 1)[-1]
        raw = netloc.rsplit(":", 1)[0] if netloc.count(":") == 1 else netloc
    return clean_host(raw)


@lru_cache(maxsize=65536)
def _registered_domain_of_host(host: str) -> str | None:
    try:
        r = _EXTRACT(host)
    except Exception as e:  # noqa: BLE001
        log.debug("tldextract(%s): %s", host, e)
        r = None
    if r is not None:
        top = r.top_domain_under_public_suffix
        if top:
            return top
        if r.suffix and not r.domain:  # host sam jest sufiksem, np. netlify.app, blogspot.com
            return host if "." in host else None
    labels = [x for x in host.split(".") if x]
    if len(labels) >= 2:  # nieznana domena najwyższego poziomu — ostatnie dwie etykiety
        return ".".join(labels[-2:])
    return None


def registered_domain(url_or_host: str | None) -> str | None:
    """Domena zarejestrowana (eTLD+1, z prywatną częścią PSL), małe litery, punycode. IP i śmieci → None."""
    host = host_of(url_or_host)
    if host is None or is_ip(host):
        return None
    return _registered_domain_of_host(host)


def _is_tracking_param(name: str) -> bool:
    n = name.strip().lower()
    return n.startswith("utm_") or n in TRACKING_PARAMS


def _netloc(parts, url: str) -> str | None:
    """host[:port] bez danych logowania i bez portu domyślnego; None, gdy host nie wygląda na adres strony."""
    host = host_of(url)
    if host is None:
        return None
    if not is_ip(host) and "." not in host:  # „brak”, „localhost”, „n/a” — to nie jest adres strony
        return None
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = f"[{host}]" if ":" in host else host
    scheme = parts.scheme.lower()
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc += f":{port}"
    return netloc


def normalize_url(url: str | None) -> str | None:
    """Adres http(s) w postaci kanonicznej: schemat i host małymi literami, punycode, bez domyślnego portu,
    bez danych logowania, fragmentu i parametrów śledzących (utm_*, fbclid, gclid, yclid, mc_eid, _ga…).
    Brak schematu → ``http://``. Inne schematy, puste i nieprawidłowe wartości → None."""
    if not url or not isinstance(url, str):
        return None
    parts = _split(url)
    if parts is None:
        return None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return None
    netloc = _netloc(parts, url)
    if netloc is None:
        return None
    path = quote(parts.path or "/", safe=_PATH_SAFE)
    query = ""
    if parts.query:
        segments = [s for s in parts.query.split("&") if s]
        kept = [s for s in segments if not _is_tracking_param(unquote_plus(s.split("=", 1)[0]))]
        query = quote("&".join(kept), safe=_QUERY_SAFE)
    return urlunsplit((scheme, netloc, path, query, ""))


def prepare_fetch_url(url: str | None) -> str | None:
    """Adres do pobrania: jak wpisany (z parametrami — część cloakerów ich wymaga), tylko przywrócony z postaci
    rozbrojonej, ze schematem i bez fragmentu. None, gdy to nie jest adres http(s)."""
    if not url or not isinstance(url, str):
        return None
    parts = _split(url)
    if parts is None or parts.scheme.lower() not in ("http", "https"):
        return None
    netloc = _netloc(parts, url)
    if netloc is None:
        return None
    return urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query, ""))


_LINK_SHIM_DOMAINS = frozenset({"facebook.com", "messenger.com", "instagram.com", "threads.net", "threads.com"})


def unwrap_redirector(url: str | None) -> str | None:
    """Adres docelowy z „opakowań” przekierowań platform (l.facebook.com/l.php?u=…, google.com/url?q=…,
    youtube.com/redirect?q=…, vk.com/away.php?to=…, Outlook SafeLinks, LinkedIn). None, gdy to nie opakowanie."""
    if not url:
        return None
    host = host_of(url)
    parts = _split(url)
    if host is None or parts is None:
        return None
    rd = registered_domain(host) or ""
    path = parts.path or "/"
    q = parse_qs(parts.query, keep_blank_values=False)

    def first(*names: str) -> str | None:
        for n in names:
            vals = q.get(n)
            if vals and vals[0].strip():
                return vals[0].strip()
        return None

    target: str | None = None
    if rd in _LINK_SHIM_DOMAINS and host.split(".")[0] in ("l", "lm"):
        target = first("u")
    elif rd.startswith("google.") and path == "/url":
        target = first("q", "url")
    elif rd == "youtube.com" and path == "/redirect":
        target = first("q")
    elif rd == "vk.com" and path == "/away.php":
        target = first("to")
    elif host.endswith("safelinks.protection.outlook.com"):
        target = first("url")
    elif rd == "linkedin.com" and path.startswith("/redir/redirect"):
        target = first("url")
    if not target:
        return None
    target = prepare_fetch_url(target)
    if target is None or target == prepare_fetch_url(url):
        return None
    return target


def is_platform_url(url: str | None) -> bool:
    """Strona platformy (Facebook, TikTok, YouTube, Telegram…) — nie pobieramy jej."""
    host = host_of(url)
    if host is None:
        return False
    if host in PLATFORM_HOSTS:
        return True
    return registered_domain(host) in PLATFORM_DOMAINS


def is_reserved_host(url_or_host: str | None) -> bool:
    """Domena specjalnego przeznaczenia (.example, .test, .invalid, .localhost, .local…) albo adres IP
    niepubliczny — takich adresów nie odpytujemy w sieci (np. zbiór syntetyczny)."""
    host = host_of(url_or_host)
    if host is None:
        return False
    if is_ip(host):
        try:
            return not ipaddress.ip_address(host).is_global
        except ValueError:
            return True
    tld = host.rsplit(".", 1)[-1]
    return tld in RESERVED_TLDS or host == "localhost" or "." not in host


def unique(values: Iterable[str | None]) -> list[str]:
    """Wartości bez powtórzeń i pustych, w kolejności pierwszego wystąpienia."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ---------------------------------------------------------------- listy wyjątków (config/stoplists.yaml)

STOPLIST_KEYS: tuple[str, ...] = ("domains", "asns", "nameserver_patterns", "registrars", "tls_issuers", "form_fields")


@lru_cache(maxsize=4)
def load_stoplists(path: str | None = None) -> dict[str, tuple[str, ...]]:
    """Listy wartości zbyt częstych, by były przesłanką (config/stoplists.yaml). Brak pliku → puste listy."""
    import yaml

    if path is None:
        from slad.settings import CONFIG_DIR

        file = CONFIG_DIR / "stoplists.yaml"
    else:
        file = Path(path)
    data: dict = {}
    try:
        loaded = yaml.safe_load(file.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded
    except FileNotFoundError:
        log.info("Brak pliku list wyjątków %s", file)
    except Exception as e:  # noqa: BLE001
        log.warning("Nie można wczytać %s: %s", file, e)
    out: dict[str, tuple[str, ...]] = {}
    for key in STOPLIST_KEYS:
        values = data.get(key) or []
        out[key] = tuple(str(v).strip().lower() for v in values if v is not None and str(v).strip())
    return out


def is_stoplisted_domain(domain_or_host: str | None, stoplists: dict[str, tuple[str, ...]] | None = None) -> bool:
    """Domena (lub host) z listy wyjątków — dopasowanie dokładne albo jako subdomena (cdn.jsdelivr.net ⊂ jsdelivr.net)."""
    host = host_of(domain_or_host)
    if not host:
        return False
    lists = stoplists if stoplists is not None else load_stoplists()
    for d in lists.get("domains", ()):
        if host == d or host.endswith("." + d):
            return True
    return False
