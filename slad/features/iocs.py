"""Wskaźniki (IOC) w tekście materiału: telefony, e-maile, URL-e i domeny, portfele kryptowalut, Telegram,
WhatsApp, IBAN.

    extract_iocs(text) -> {"phone": [...], "email": [...], "url": [...], "domain": [...], "crypto_btc": [...],
                           "crypto_eth": [...], "crypto_trx": [...], "telegram": [...], "whatsapp": [...],
                           "iban": [...]}

Wszystkie klucze są zawsze obecne. Wartości są znormalizowane, bez duplikatów, w kolejności wystąpienia:
telefon E.164 („+48XXXXXXXXX”), e-mail małymi literami, URL z protokołem, domena zarejestrowana (eTLD+1),
BTC/TRX bez zmian (wielkość liter ma znaczenie), ETH małymi literami, Telegram „@nazwa” (zaproszenia: „t.me/+kod”),
WhatsApp „+cyfry” (tylko numery; zaproszenia do grup trafiają do „url”), IBAN wielkimi literami bez spacji.

Precyzja ważniejsza niż czułość — fałszywy wskaźnik to fałszywe powiązanie dwóch kampanii:
- daty, kwoty, numery kont, NIP/REGON i identyfikatory zamówień nie są telefonami;
- adresy platform (facebook.com, t.me, bit.ly, gov.pl, portale informacyjne, banki…) nie są domenami kampanii;
  na hostingach (wordpress.com, netlify.app…) domeną jest cała nazwa hosta;
- adresy BTC/TRX: suma kontrolna Base58Check, a bez niej tylko w kontekście kryptowalut; IBAN: suma mod 97;
- zapisy „rozbrojone” (hxxp://, przyklad[.]pl, kontakt[at]przyklad.pl) są rozpoznawane.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

log = logging.getLogger(__name__)

IOC_KINDS = (
    "phone",
    "email",
    "url",
    "domain",
    "crypto_btc",
    "crypto_eth",
    "crypto_trx",
    "telegram",
    "whatsapp",
    "iban",
)
MAX_TEXT = 200_000  # dłuższe teksty są przycinane (czas działania wyrażeń regularnych)

# ------------------------------------------------------------------ domeny

# Hostingi i kreatory stron: wspólna domena zarejestrowana nie łączy kampanii — liczy się pełna nazwa hosta.
HOSTING_DOMAINS = frozenset(
    """
    github.io gitlab.io netlify.app vercel.app herokuapp.com blogspot.com web.app firebaseapp.com pages.dev
    workers.dev glitch.me wixsite.com weebly.com wordpress.com 000webhostapp.com azurewebsites.net
    appspot.com onrender.com cloudfront.net replit.app repl.co carrd.co square.site webflow.io tilda.ws
    mystrikingly.com godaddysites.com business.site framer.website site123.me jimdosite.com
    """.split()
)

# Sufiksy, pod którymi rejestruje się domeny 3. poziomu — uproszczona Public Suffix List, używana tylko wtedy,
# gdy moduł slad.features.infra.domains jest niedostępny. Hostingi traktujemy tak samo (subdomena = osobna strona).
MULTI_LABEL_SUFFIXES = HOSTING_DOMAINS | frozenset(
    """
    com.pl net.pl org.pl info.pl biz.pl edu.pl gov.pl mil.pl nom.pl priv.pl waw.pl warszawa.pl krakow.pl
    wroclaw.pl poznan.pl gda.pl gdansk.pl lodz.pl katowice.pl szczecin.pl lublin.pl bydgoszcz.pl torun.pl
    olsztyn.pl opole.pl rzeszow.pl bialystok.pl kielce.pl slask.pl auto.pl sklep.pl shop.pl media.pl tm.pl
    pc.pl turystyka.pl travel.pl
    co.uk org.uk ac.uk gov.uk me.uk ltd.uk plc.uk com.au net.au org.au co.nz co.jp co.kr co.in co.za co.il
    com.br com.ar com.mx com.tr com.ua com.cn com.hk com.sg com.my com.ph com.vn com.ng com.eg com.sa
    """.split()
)

# Platformy, skracacze linków, infrastruktura WWW i podszywane (prawdziwe) serwisy: nigdy nie są „domeną kampanii”.
# Adres URL na takiej domenie zostaje wskaźnikiem tylko wtedy, gdy ma ścieżkę (np. bit.ly/abc, facebook.com/strona).
PLATFORM_DOMAINS = frozenset(
    """
    facebook.com fb.com fb.me fb.watch messenger.com instagram.com threads.net whatsapp.com wa.me t.me
    telegram.me telegram.org telegram.dog google.com google.pl youtube.com youtu.be tiktok.com twitter.com x.com
    linkedin.com pinterest.com reddit.com snapchat.com wikipedia.org apple.com microsoft.com live.com
    gov.pl europa.eu w3.org schema.org gstatic.com googleapis.com googletagmanager.com google-analytics.com
    doubleclick.net googlesyndication.com cloudflare.com jquery.com bootstrapcdn.com jsdelivr.net gravatar.com
    wordpress.org bit.ly tinyurl.com cutt.ly t.co goo.gl ow.ly is.gd rebrand.ly shorturl.at tiny.cc rb.gy
    linktr.ee buff.ly lnkd.in discord.com discord.gg vk.com ok.ru viber.com github.com icloud.com outlook.com
    msn.com bing.com yandex.ru yandex.com wix.com
    onet.pl wp.pl interia.pl o2.pl tvn24.pl tvn.pl polsatnews.pl polsat.pl tvp.pl tvp.info rp.pl wyborcza.pl
    fakt.pl se.pl pb.pl money.pl bankier.pl businessinsider.com.pl forbes.pl pudelek.pl rmf24.pl rmf.fm
    radiozet.pl gazeta.pl natemat.pl
    pkobp.pl pko.pl ipko.pl pekao.com.pl mbank.pl ing.pl santander.pl bankmillennium.pl aliorbank.pl
    credit-agricole.pl bnpparibas.pl revolut.com nbp.pl knf.gov.pl zus.pl orlen.pl pge.pl pgnig.pl kghm.com
    pzu.pl gpw.pl allegro.pl olx.pl inpost.pl poczta-polska.pl dhl.com dpd.com.pl vinted.pl binance.com
    coinbase.com tesla.com amazon.com netflix.com biedronka.pl lidl.pl lotto.pl
    """.split()
)

_FILE_EXTENSIONS = frozenset("png jpg jpeg gif webp svg bmp ico js css pdf mp4 webm json xml".split())


def _infra(name: str) -> Callable[..., Any] | None:
    """Funkcja z slad.features.infra.domains (spójna normalizacja z modułem infrastruktury), jeśli istnieje."""
    if name not in _INFRA:
        try:
            from slad.features.infra import domains as _domains

            _INFRA[name] = getattr(_domains, name, None)
        except Exception as e:  # noqa: BLE001 — moduł infrastruktury jest opcjonalny
            log.debug("slad.features.infra.domains niedostępny (%s) — używam wersji uproszczonej", e)
            _INFRA[name] = None
    return _INFRA[name]


_INFRA: dict[str, Callable[..., Any] | None] = {}


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def fallback_registered_domain(host: str) -> str:
    """Domena zarejestrowana bez Public Suffix List: 2 ostatnie etykiety (3 dla com.pl, co.uk, hostingów…)."""
    host = (host or "").strip().strip(".").lower()
    if not host or _is_ip(host):
        return ""
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if ".".join(labels[-2:]) in MULTI_LABEL_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def registered_domain(host: str) -> str:
    """eTLD+1 hosta: slad.features.infra.domains.registered_domain, a gdy niedostępna — wersja uproszczona."""
    host = (host or "").strip().strip(".").lower()
    if not host or _is_ip(host):
        return ""
    fn = _infra("registered_domain")
    if fn is not None:
        try:
            rd = fn(host)
            if isinstance(rd, str) and rd.strip():
                return rd.strip().lower()
        except Exception as e:  # noqa: BLE001
            log.debug("registered_domain(%s) z infra: %s", host, e)
    return fallback_registered_domain(host)


# ------------------------------------------------------------------ wyrażenia

_REFANG = (
    (re.compile(r"\bhxxp(s?)(?=://)", re.I), r"http\1"),
    (re.compile(r"\s?(?:\[\.\]|\(\.\)|\{\.\}|\[dot\]|\(dot\))\s?", re.I), "."),
    (re.compile(r"(?<=\w)\s?(?:\[at\]|\(at\)|\[@\])\s?(?=\w)", re.I), "@"),
)

_EMAIL = re.compile(
    r"(?<![\w.%+-])[A-Za-z0-9][A-Za-z0-9._%+-]{0,63}@"
    r"((?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24})(?![\w-])"
)

_URL_CHARS = r"[^\s<>\"'`{}|\\^\[\]]"
_URL_SCHEME = re.compile(rf"(?<![\w.-])https?://{_URL_CHARS}+", re.I)
_URL_WWW = re.compile(rf"(?<![\w./@-])www\d{{0,3}}\.{_URL_CHARS}+", re.I)
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
BARE_TLDS = ("pl", "com", "net", "org", "info", "io", "app", "online", "site", "top", "shop", "store", "click")
BARE_TLDS += ("xyz", "live", "pro", "biz", "eu")
_URL_BARE = re.compile(
    rf"(?<![\w./@-])(?:{_LABEL}\.)+(?:{'|'.join(BARE_TLDS)})(?![\w-])(?:/{_URL_CHARS}*)?",
    re.I,
)
_URL_SHORT = re.compile(  # skracacze linków pisane bez protokołu (TLD spoza listy BARE_TLDS)
    r"(?<![\w./@-])(?:bit\.ly|cutt\.ly|rb\.gy|is\.gd|ow\.ly|shorturl\.at|tiny\.cc|rebrand\.ly|goo\.gl|s\.id)"
    r"/[A-Za-z0-9_-]{3,}",
    re.I,
)
_TRAILING = ".,;:!?…»«\"'”’“)]}>*"
_TRACKING_PARAM = re.compile(
    r"^(?:utm_\w*|fbclid|gclid|gbraid|wbraid|dclid|yclid|msclkid|ttclid|twclid|mc_eid|_ga|_gl)$", re.I
)
_HOST_OK = re.compile(r"^[\w.-]+$")

_TG_LINK = re.compile(
    r"(?<![\w.-])(?:https?://)?(?:www\.)?(?:t|telegram)\.(?:me|dog)/(?:s/)?"
    r"(\+[\w-]{5,}|joinchat/[\w-]{5,}|[A-Za-z][A-Za-z0-9_]{3,31})(?!\w)",
    re.I,
)
_TG_RESERVED = frozenset(
    "share addstickers addemoji addtheme addlist proxy socks iv login setlanguage confirmphone boost invoice contact"
    " auth joinchat".split()
)
_AT_HANDLE = re.compile(r"(?<![\w@./])@([A-Za-z][A-Za-z0-9_]{4,31})(?![\w@])")
_TELEGRAM_WORD = re.compile(r"telegram", re.I)

_WA_LINK = re.compile(
    r"(?<![\w.-])(?:https?://)?(?:www\.)?"
    r"(?:wa\.me/|(?:api\.|web\.)?whatsapp\.com/send/?\?(?:[^\s#]*?&)?phone=)(?:%2B|\+)?(\d{6,15})(?!\d)",
    re.I,
)
_WA_GROUP = re.compile(r"(?<![\w.-])(?:https?://)?chat\.whatsapp\.com/(?:invite/)?([A-Za-z0-9]{10,40})(?!\w)", re.I)

_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {c: i for i, c in enumerate(_B58_ALPHABET)}
_BTC_LEGACY = re.compile(r"(?<![A-Za-z0-9])[13][1-9A-HJ-NP-Za-km-z]{25,34}(?![A-Za-z0-9])")
_BTC_BECH32 = re.compile(r"(?<![A-Za-z0-9])bc1[02-9ac-hj-np-z]{39,59}(?![A-Za-z0-9])", re.I)
_ETH = re.compile(r"(?<![A-Za-z0-9])0x[0-9a-f]{40}(?![0-9a-z])", re.I)
_TRX = re.compile(r"(?<![A-Za-z0-9])T[1-9A-HJ-NP-Za-km-z]{33}(?![A-Za-z0-9])")
_CRYPTO_CONTEXT = re.compile(
    r"bitcoin|\bbtc\b|krypto|crypto|portfel|wallet|usdt|tether|\btron|trc-?20|erc-?20|ethereum|\beth\b|blockchain|"
    r"binance|coin",
    re.I,
)

# Długości IBAN (ISO 13616) — tylko kraje z tej tabeli; PL dodatkowo bez prefiksu kraju (NRB, 26 cyfr).
IBAN_LENGTHS = {
    "PL": 28, "DE": 22, "GB": 22, "LT": 20, "LV": 21, "EE": 20, "CZ": 24, "SK": 24, "FR": 27, "ES": 24,
    "IT": 27, "NL": 18, "BE": 16, "AT": 20, "CH": 21, "IE": 22, "CY": 28, "MT": 31, "UA": 29, "BG": 22,
    "RO": 24, "HU": 28, "PT": 25, "LU": 20, "FI": 18, "SE": 24, "DK": 18, "NO": 15, "HR": 21, "SI": 19,
    "GR": 27, "GE": 22, "TR": 26, "AE": 23, "GI": 23, "IS": 26, "LI": 21, "MC": 27, "AD": 24, "AL": 28,
    "BA": 20, "ME": 22, "MK": 19, "RS": 22, "MD": 24, "BY": 28, "AZ": 28, "KZ": 20, "IL": 23, "SA": 24,
}  # fmt: skip
_IBAN_START = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2})(\d{2})")
_NRB = re.compile(r"(?<![\w])(\d{2}) ?(\d{4}) ?(\d{4}) ?(\d{4}) ?(\d{4}) ?(\d{4}) ?(\d{4})(?![\w])")

# Telefony: ciąg grup cyfr rozdzielonych spacją / myślnikiem (opcjonalnie +, 00 i nawiasy wokół grupy)
_PHONE_RUN = re.compile(
    r"(?<![\w+/=%])(?<!\d[.,-])"
    r"(?:\(\s?)?(?:\+|00)?\d+\)?"
    r"(?:[ \u00a0\u2009\u202f-]{1,2}\(?\d+\)?)*"
    r"(?![\w%/]|[.,]\d)"
)
_AMOUNT_AFTER = re.compile(
    r"\s*(?:zł|zl\b|pln|złot|zlot|euro?\b|€|\$|usd|dolar|tys\b|tys\.|mln|mld|%|proc|osób|osob|razy|sztuk|szt\b|"
    r"km\b|kg\b)",
    re.I,
)
_ID_BEFORE = re.compile(
    r"(?:regon|nip|krs|pesel|konta|konto|rachun\w*|iban|faktur\w*|zam[oó]wieni\w*|przesy[lł]k\w*|paczk\w*|"
    r"sygn\w*|ref\w*|\bid|kod\w*|order|invoice|tracking|transakcj\w*|umow\w*|polis\w*|klient\w*)"
    r"\W{0,4}(?:nr|no|numer)?\.?\W{0,3}$",
    re.I,
)
_PHONE_BEFORE = re.compile(
    r"(?:tel|telefon\w*|zadzwo\w*|dzwo\w*|kontakt\w*|infolini\w*|sms\w*|whats\s?app|kom[oó]rk\w*|numer\w*|\bnr|"
    r"phone|call|mobile)\W{0,6}$",
    re.I,
)
PL_MOBILE_PREFIXES = frozenset("45 50 51 53 57 60 66 69 72 73 78 79 88".split())
PL_PREFIXES = PL_MOBILE_PREFIXES | frozenset(
    "12 13 14 15 16 17 18 22 23 24 25 29 32 33 34 41 42 43 44 46 48 52 54 55 56 58 59 61 62 63 65 67 68 71 74 75"
    " 76 77 81 82 83 84 85 86 87 89 91 94 95 39 70 80".split()
)

_Found = dict[str, list[tuple[int, str]]]


# ------------------------------------------------------------------ API


def extract_iocs(text: str) -> dict[str, list[str]]:
    """Wskaźniki z tekstu (patrz opis modułu). Nigdy nie rzuca wyjątku — w razie błędu zwraca to, co zebrało."""
    found: _Found = {k: [] for k in IOC_KINDS}
    if isinstance(text, str) and text.strip():
        try:
            _extract(text[:MAX_TEXT], found)
        except Exception:  # noqa: BLE001 — wskaźniki nie mogą zatrzymać przetwarzania materiału
            log.warning("Ekstrakcja IOC przerwana", exc_info=True)
    return {kind: _dedupe(items) for kind, items in found.items()}


def _dedupe(items: list[tuple[int, str]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for _, value in sorted(items, key=lambda x: x[0]):
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _extract(text: str, found: _Found) -> None:
    work = text
    for rx, repl in _REFANG:
        work = rx.sub(repl, work)
    chars = list(work)

    def blank(start: int, end: int) -> None:
        chars[start:end] = " " * (end - start)

    def view() -> str:
        return "".join(chars)

    # 1. e-maile (zanim domeny i @nazwy)
    for m in _EMAIL.finditer(work):
        if m.group(1).rsplit(".", 1)[-1].lower() in _FILE_EXTENSIONS:
            continue
        found["email"].append((m.start(), m.group(0).lower().rstrip(".")))
        blank(*m.span())
    cur = view()

    # 2. komunikatory (osobne klucze — nie trafiają do url/domain)
    for m in _TG_LINK.finditer(cur):
        handle = m.group(1)
        if handle.startswith("+"):
            value = "t.me/" + handle
        elif handle.lower().startswith("joinchat/"):
            value = "t.me/+" + handle.split("/", 1)[1]
        elif handle.lower() in _TG_RESERVED:
            continue
        else:
            value = "@" + handle.lower()
        found["telegram"].append((m.start(), value))
        blank(m.start(), _token_end(cur, m.end()))
    for m in _WA_LINK.finditer(cur):
        digits = m.group(1)
        found["whatsapp"].append((m.start(), "+" + ("48" + digits if len(digits) == 9 else digits)))
        blank(m.start(), _token_end(cur, m.end()))
    for m in _WA_GROUP.finditer(cur):  # zaproszenie do grupy to nie numer — wskaźnik jako URL (kod jest unikalny)
        found["url"].append((m.start(), "https://chat.whatsapp.com/" + m.group(1)))
        blank(m.start(), _token_end(cur, m.end()))
    pre_url = view()

    # 3. adresy URL: z protokołem → www. → same domeny z popularnymi TLD
    url_spans: list[tuple[int, int]] = []
    cur = pre_url
    for rx in (_URL_SCHEME, _URL_WWW, _URL_SHORT, _URL_BARE):
        for m in rx.finditer(cur):
            raw = _strip_trailing(m.group(0))
            if not raw:
                continue
            end = m.start() + len(raw)
            if _add_url(raw, m.start(), found):
                url_spans.append((m.start(), end))
            blank(m.start(), end)
        cur = view()

    # 4. @nazwa w Telegramie — tylko gdy słowo „telegram” stoi tuż przed („Telegram: @x”) albo tuż za („@x na Telegramie”)
    for m in _AT_HANDLE.finditer(cur):
        before, after = work[max(0, m.start() - 60) : m.start()], work[m.end() : m.end() + 30]
        if _TELEGRAM_WORD.search(before) or _TELEGRAM_WORD.search(after):
            found["telegram"].append((m.start(), "@" + m.group(1).lower()))
            blank(*m.span())

    # 5. kryptowaluty (na tekście z adresami URL — portfel bywa w linku do eksploratora)
    has_crypto_context = bool(_CRYPTO_CONTEXT.search(work))
    for m in _BTC_BECH32.finditer(pre_url):
        s = m.group(0)
        if s == s.lower() or s == s.upper():
            found["crypto_btc"].append((m.start(), s.lower()))
            blank(*m.span())
    for m in _ETH.finditer(pre_url):
        found["crypto_eth"].append((m.start(), m.group(0).lower()))
        blank(*m.span())
    for rx, kind, versions in ((_BTC_LEGACY, "crypto_btc", (0x00, 0x05)), (_TRX, "crypto_trx", (0x41,))):
        for m in rx.finditer(pre_url):
            s = m.group(0)
            in_url = any(a <= m.start() < b for a, b in url_spans)
            if _base58check(s, versions) or (has_crypto_context and not in_url and _looks_like_address(s)):
                found[kind].append((m.start(), s))
                blank(*m.span())
    cur = view()

    # 6. IBAN (przed telefonami — numer konta nie może dać „telefonu”)
    for start, end, iban in _find_ibans(cur):
        found["iban"].append((start, iban))
        blank(start, end)
    cur = view()

    # 7. telefony
    for start, phone in _find_phones(cur):
        found["phone"].append((start, phone))


def _token_end(text: str, i: int) -> int:
    while i < len(text) and not text[i].isspace() and text[i] not in "<>\"'":
        i += 1
    return i


def _strip_trailing(u: str) -> str:
    while u and u[-1] in _TRAILING:
        if u[-1] == ")" and u.count("(") >= u.count(")"):
            break
        u = u[:-1]
    return u


def normalize_url(raw: str) -> str | None:
    """URL z tekstu → postać kanoniczna (protokół, host małymi literami, bez fragmentu i domyślnego portu).
    Gdy dostępna, stosowana jest też normalizacja modułu infrastruktury (spójność z InfraFeatures)."""
    u = _strip_trailing(raw.strip())
    if not u:
        return None
    if not re.match(r"^[a-z][a-z0-9+.-]*://", u, re.I):
        u = "http://" + u
    try:
        parts = urlsplit(u)
        host = (parts.hostname or "").rstrip(".").lower()
        port = parts.port
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not host or not _HOST_OK.match(host):
        return None
    if not _is_ip(host):
        tld = host.rsplit(".", 1)[-1]
        if "." not in host or not (tld.isalpha() or tld.startswith("xn--")) or len(tld) < 2:
            return None
    if not host.isascii():  # IDN → punycode, jak w slad.features.infra.domains
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError:
            return None
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"
    query = "&".join(
        p for p in parts.query.split("&") if p and not _TRACKING_PARAM.match(p.split("=", 1)[0])
    )  # utm_*, fbclid… identyfikują kliknięcie, nie kampanię
    norm = urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", query, ""))
    fn = _infra("normalize_url")
    if fn is not None:
        try:
            out = fn(norm)
            if isinstance(out, str) and out.strip():
                return out.strip()
        except Exception as e:  # noqa: BLE001
            log.debug("normalize_url(%s) z infra: %s", norm, e)
    return norm


def _add_url(raw: str, pos: int, found: _Found) -> bool:
    url = normalize_url(raw)
    if not url:
        return False
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    plain_host = host.removeprefix("www.")
    rd = registered_domain(host)
    has_path = parts.path not in ("", "/") or bool(parts.query)
    if rd in PLATFORM_DOMAINS or plain_host in PLATFORM_DOMAINS:
        if has_path:
            found["url"].append((pos, url))
        return True
    found["url"].append((pos, url))
    if _is_ip(host):
        return True
    if rd in HOSTING_DOMAINS:  # wspólny hosting ≠ wspólna kampania: liczy się konkretna strona (etykieta pod nim)
        if plain_host.endswith("." + rd):
            found["domain"].append((pos, plain_host[: -len(rd) - 1].rsplit(".", 1)[-1] + "." + rd))
    elif rd:
        found["domain"].append((pos, rd))
    return True


def _base58check(s: str, versions: tuple[int, ...]) -> bool:
    n = 0
    for ch in s:
        idx = _B58_INDEX.get(ch)
        if idx is None:
            return False
        n = n * 58 + idx
    pad = len(s) - len(s.lstrip("1"))
    raw = b"\x00" * pad + (n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b"")
    if len(raw) != 25 or raw[0] not in versions:
        return False
    return hashlib.sha256(hashlib.sha256(raw[:-4]).digest()).digest()[:4] == raw[-4:]


def _looks_like_address(s: str) -> bool:
    """Losowo wyglądający ciąg Base58 (cyfry + wielkie + małe litery) — dla adresów bez poprawnej sumy kontrolnej
    (np. zmyślonych w zbiorze syntetycznym albo z błędem OCR), dopuszczanych tylko w kontekście kryptowalut."""
    digits = sum(c.isdigit() for c in s)
    upper = sum(c.isupper() for c in s)
    lower = sum(c.islower() for c in s)
    return digits >= 2 and upper >= 3 and lower >= 3


def iban_valid(iban: str) -> bool:
    s = iban.replace(" ", "").upper()
    if len(s) < 15 or not s[:2].isalpha() or not s[2:4].isdigit() or not s.isalnum():
        return False
    rearranged = s[4:] + s[:4]
    try:
        return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1
    except ValueError:
        return False


def _find_ibans(text: str) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for m in _IBAN_START.finditer(text):
        cc = m.group(1)
        length = IBAN_LENGTHS.get(cc)
        if not length:
            continue
        chars = [cc, m.group(2)]
        need, i = length - 4, m.end()
        while need and i < len(text):
            c = text[i]
            if c == " " and i + 1 < len(text) and text[i + 1].isascii() and text[i + 1].isalnum():
                i += 1
                continue
            if not (c.isascii() and c.isalnum()):
                break
            chars.append(c.upper())
            need -= 1
            i += 1
        if need or (i < len(text) and text[i].isascii() and text[i].isalnum()):
            continue
        iban = "".join(chars)
        if cc == "PL" and not iban[2:].isdigit():
            continue
        if iban_valid(iban):
            out.append((m.start(), i, iban))
    taken = [(s, e) for s, e, _ in out]
    for m in _NRB.finditer(text):  # polski numer rachunku bez „PL”
        if any(s <= m.start() < e for s, e in taken):
            continue
        iban = "PL" + "".join(m.groups())
        if iban_valid(iban):
            out.append((m.start(), m.end(), iban))
    return out


def _find_phones(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for m in _PHONE_RUN.finditer(text):
        raw, start, end = m.group(0), m.start(), m.end()
        groups = re.findall(r"\d+", raw)
        head = raw.lstrip("( \u00a0")
        if _AMOUNT_AFTER.match(text, end):
            continue
        if head.startswith(("+", "00")):
            phone = _international(groups, head.startswith("00"))
            if phone:
                out.append((start, phone))
            continue
        if _ID_BEFORE.search(text[max(0, start - 40) : start]):
            continue
        phone_context = bool(_PHONE_BEFORE.search(text[max(0, start - 30) : start]))
        i = 0
        while i < len(groups):  # okna 9 cyfr wyrównane do grup, bez nakładania
            acc, j = "", i
            while j < len(groups) and len(acc) < 9:
                acc += groups[j]
                j += 1
            compact = j - i == 1
            if len(acc) == 9 and _valid_pl(acc) and (not compact or acc[:2] in PL_MOBILE_PREFIXES or phone_context):
                out.append((start, "+48" + acc))
                i = j
            else:
                i += 1
    return out


def _valid_pl(national: str) -> bool:
    return len(national) == 9 and national[:2] in PL_PREFIXES and "00000" not in national


def _international(groups: list[str], from_00: bool) -> str | None:
    groups = list(groups)
    if from_00:
        groups[0] = groups[0][2:]
    groups = [g for g in groups if g]
    joined = "".join(groups)
    if not joined or joined[0] == "0":
        return None
    if joined.startswith("48"):
        first = groups[0]
        rest = ([first[2:]] if len(first) > 2 else []) + groups[1:]
        acc = ""
        for g in rest:
            if len(acc) >= 9:
                break
            acc += g
        if len(acc) == 9 and acc[0] != "0" and "00000" not in acc:
            return "+48" + acc
        return None
    if 8 <= len(joined) <= 15 and "00000" not in joined[1:]:
        return "+" + joined
    return None
