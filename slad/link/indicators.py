"""Dyskretne wskaźniki materiału: wartości do dokładnego dopasowania i do indeksu odwrotnego.

`indicators(fs)` → {rodzaj: {znormalizowane wartości}}. Rodzaje (klucze wag z config/settings.yaml → link.weights):

  infrastruktura: tracker_id, crypto_wallet, phone, email, telegram, whatsapp, iban, tls_fingerprint,
                  cert_domain, domain, host, short_url, script_domain, favicon, form_action, form_fields,
                  ip, asn, nameserver, title
  treść / osoby:  person (nazwiska z tekstu + tożsamości twarzy), brand, narrative, keyword

Zasady odporności (fałszywe powiązania kosztują najwięcej):
- wartości z config/stoplists.yaml + wbudowane listy (platformy, CDN, wspólne serwery DNS) są pomijane,
- strona-wizytówka platformy (np. przekierowanie na facebook.com), strona błędu, parking domeny czy
  wyzwanie Cloudflare („Just a moment…”) NIE dostarczają cech strony (tytuł, favicon, szablon, formularz),
- adresy prywatne / zastrzeżone i adresy brzegowe CDN nie są wskaźnikiem IP,
- domeny na współdzielonych hostingach (wixsite.com, netlify.app, …) są identyfikowane po pełnym hoście.
"""

from __future__ import annotations

import functools
import ipaddress
import logging
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from slad.models import FeatureSet

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- metoda: grupy, wagi domyślne, rodzaje miękkie

# rodzaj przesłanki → grupa (models.GROUPS)
KIND_GROUP: dict[str, str] = {
    "tracker_id": "infrastruktura",
    "crypto_wallet": "infrastruktura",
    "phone": "infrastruktura",
    "email": "infrastruktura",
    "telegram": "infrastruktura",
    "whatsapp": "infrastruktura",
    "iban": "infrastruktura",
    "tls_fingerprint": "infrastruktura",
    "cert_domain": "infrastruktura",
    "domain": "infrastruktura",
    "redirect_domain": "infrastruktura",
    "host": "infrastruktura",
    "short_url": "infrastruktura",
    "script_domain": "infrastruktura",
    "favicon": "infrastruktura",
    "form_action": "infrastruktura",
    "form_fields": "infrastruktura",
    "form_offer": "infrastruktura",
    "ip": "infrastruktura",
    "asn": "infrastruktura",
    "nameserver": "infrastruktura",
    "title": "infrastruktura",
    "dom_simhash": "infrastruktura",
    "registrar_created": "infrastruktura",
    "text_minhash": "tekst",
    "text_tfidf": "tekst",
    "keyword": "tekst",
    "image_phash": "obraz",
    "image_clip": "obraz",
    "face": "osoby",
    "person": "osoby",
    "brand": "narracja",
    "narrative": "narracja",
}

# Wagi używane, gdy config/settings.yaml (link.weights) nie zawiera danego rodzaju.
DEFAULT_WEIGHTS: dict[str, float] = {
    "tracker_id": 6.0,
    "crypto_wallet": 7.0,
    "phone": 6.0,
    "email": 6.0,
    "iban": 6.0,  # jak telefon: identyfikator operatora
    "telegram": 5.0,
    "whatsapp": 5.0,
    "tls_fingerprint": 5.0,
    "cert_domain": 4.0,  # domena z tego samego certyfikatu TLS / CT
    "domain": 5.0,
    "short_url": 5.0,  # ten sam skrócony link (bit.ly/…)
    "redirect_domain": 3.0,  # domena pośrednia łańcucha przekierowań (cloaker / TDS / skracacz)
    "favicon": 4.0,
    "form_action": 4.0,
    "host": 3.0,
    "script_domain": 3.0,  # skrypty z tej samej (niepublicznej) domeny
    "dom_simhash": 3.0,
    "ip": 2.5,
    "title": 2.0,
    "form_fields": 2.0,  # nietypowy zestaw pól formularza (zestaw „kit”)
    "form_offer": 3.0,  # ukryte pole formularza z identyfikatorem oferty / strumienia sieci CPA
    "registrar_created": 1.5,
    "nameserver": 1.5,
    "asn": 0.5,
    "text_minhash": 5.0,
    "text_tfidf": 3.0,
    "keyword": 0.5,
    "image_phash": 5.0,
    "image_clip": 3.0,
    "face": 4.0,
    "person": 2.0,
    "brand": 1.0,
    "narrative": 1.0,
}

# Przesłanki „miękkie”: wspólne dla wielu niezależnych kampanii (ta sama twarz celebryty, ta sama narracja,
# popularny dostawca DNS). Same nie mogą powiązać materiałów — patrz evidence.py (limit przesłanek miękkich).
SOFT_KINDS = frozenset(
    {"person", "face", "brand", "narrative", "keyword", "asn", "nameserver", "registrar_created", "image_clip"}
)

# Przesłanki „silne”: identyfikatory operatora — jedna wystarcza do powiązania (jeśli waga po rzadkości ≥ próg).
# Pozostałe twarde (host, favicon, IP, tytuł, formularz, domena pośrednia, szablon…) to przesłanki „średnie”:
# powiązanie wymaga przesłanki silnej albo co najmniej dwóch różnych rodzajów średnich (evidence.py).
STRONG_KINDS = frozenset(
    {
        "tracker_id",
        "crypto_wallet",
        "phone",
        "email",
        "telegram",
        "whatsapp",
        "iban",
        "tls_fingerprint",
        "cert_domain",  # domena z tego samego certyfikatu TLS (SAN / CT) — certyfikat zamawia operator
        "domain",
        "short_url",
    }
)

# Ile najsilniejszych wspólnych wartości jednego rodzaju liczy się do wyniku (domyślnie 2; druga z połową wagi)
MAX_VALUES: dict[str, int] = {
    "nameserver": 1,
    "asn": 1,
    "title": 1,
    "favicon": 1,
    "tls_fingerprint": 1,
    "form_fields": 1,
}

# Rodzaje wskaźników (kolejność = kolejność wyświetlania / iteracji)
INDICATOR_KINDS: tuple[str, ...] = (
    "tracker_id",
    "crypto_wallet",
    "phone",
    "email",
    "telegram",
    "whatsapp",
    "iban",
    "tls_fingerprint",
    "cert_domain",
    "domain",
    "redirect_domain",
    "host",
    "short_url",
    "script_domain",
    "favicon",
    "form_action",
    "form_fields",
    "form_offer",
    "ip",
    "asn",
    "nameserver",
    "title",
    "person",
    "brand",
    "narrative",
    "keyword",
)

# Rodzaje używane w indeksie odwrotnym kandydatów (miękkie i tak nie wystarczą do powiązania)
INDEX_KINDS: tuple[str, ...] = tuple(k for k in INDICATOR_KINDS if k not in SOFT_KINDS)


def weight(weights: dict[str, float] | None, kind: str) -> float:
    """Waga rodzaju przesłanki: konfiguracja → wartość domyślna (nieznany rodzaj → 1.0)."""
    if weights and kind in weights:
        try:
            return float(weights[kind])
        except (TypeError, ValueError):
            pass
    return DEFAULT_WEIGHTS.get(kind, 1.0)


# ---------------------------------------------------------------- listy wbudowane (uzupełniane config/stoplists.yaml)

# Platformy i duże serwisy legalne: domena nie jest śladem operatora kampanii (podszywane media, banki, CDN…).
_DEFAULT_STOP_DOMAINS = """
facebook.com fb.com fb.me fb.watch fbcdn.net facebook.net messenger.com instagram.com cdninstagram.com
whatsapp.com whatsapp.net wa.me t.me telegram.org telegram.me telegram.dog
google.com google.pl googleapis.com gstatic.com googleusercontent.com googletagmanager.com
google-analytics.com doubleclick.net googlesyndication.com googleadservices.com withgoogle.com goo.gl
youtube.com youtu.be ytimg.com tiktok.com tiktokcdn.com tiktokv.com twitter.com x.com twimg.com t.co
linkedin.com lnkd.in pinterest.com reddit.com wikipedia.org wikimedia.org apple.com icloud.com
microsoft.com live.com outlook.com office.com bing.com msn.com yahoo.com amazon.com amazonaws.com
cloudfront.net cloudflare.com cloudflare.net cloudflareinsights.com jsdelivr.net cdnjs.com unpkg.com
jquery.com bootstrapcdn.com fontawesome.com gravatar.com wordpress.org w.org wp.com
yandex.ru yandex.com yandex.net vk.com mail.ru gmail.com hotmail.com proton.me protonmail.com
onet.pl wp.pl interia.pl o2.pl gazeta.pl tvn24.pl tvn.pl polsatnews.pl polsat.pl rmf24.pl rmf.fm
tvp.pl tvp.info money.pl bankier.pl businessinsider.com.pl forbes.pl rp.pl wyborcza.pl fakt.pl se.pl
pudelek.pl natemat.pl dziennik.pl wprost.pl newsweek.pl gov.pl nbp.pl knf.gov.pl cert.pl nask.pl
pkobp.pl ipko.pl mbank.pl ing.pl ingbank.pl santander.pl pekao.com.pl bankmillennium.pl aliorbank.pl
credit-agricole.pl revolut.com paypal.com blik.com orlen.pl pge.pl pgnig.pl pzu.pl inpost.pl
allegro.pl olx.pl poczta-polska.pl zus.pl binance.com coinbase.com etoro.com xtb.com bitcoin.org
sedoparking.com parkingcrew.net bodis.com dan.com afternic.com hugedomains.com godaddy.com sav.com
namecheap.com above.com
example.com example.org example.net
"""

# Skracacze linków: sama domena nic nie znaczy, ale ten sam skrócony link (host + ścieżka) — bardzo wiele.
_SHORTENERS = """
bit.ly bitly.com tinyurl.com t.co goo.gl ow.ly is.gd buff.ly cutt.ly rb.gy shorturl.at tiny.cc rebrand.ly
s.id v.gd t.ly lnkd.in fb.me linktr.ee qrco.de bl.ink short.gy soo.gd clck.ru u.to shorturl.com
"""

# Współdzielone hostingi / kreatory stron: tożsamość witryny = pełny host (foo.wixsite.com), nie domena.
_HOSTING_DOMAINS = """
wixsite.com wix.com editorx.io wixstudio.io weebly.com wordpress.com blogspot.com blogger.com tilda.ws
webflow.io carrd.co glitch.me square.site mystrikingly.com strikingly.com jimdosite.com jimdofree.com
godaddysites.com site123.me wpcomstaging.com netlify.app vercel.app github.io gitlab.io pages.dev
workers.dev web.app firebaseapp.com herokuapp.com onrender.com azurewebsites.net appspot.com repl.co
replit.app surge.sh 000webhostapp.com translate.goog notion.site framer.website framer.app myshopify.com
webnode.page webnode.pl cba.pl prv.pl ngrok.io ngrok-free.app hostingersite.com
"""

# Systemy autonomiczne CDN / wielkich platform (jeden ASN = tysiące niezależnych witryn)
_DEFAULT_STOP_ASNS = "13335 209242 16509 14618 15169 396982 8075 20940 16625 54113 32934 13414 36459"

# Wspólne serwery DNS dostawców (wzorce: glob, podciąg lub wyrażenie regularne)
_DEFAULT_NS_PATTERNS = [
    "cloudflare",  # para serwerów nazw Cloudflare (*.ns.cloudflare.com) — wspólna dla tysięcy kont
    "*.ns.cloudflare.com",
    "*.cloudflare.com",
    "*.domaincontrol.com",
    "*.registrar-servers.com",
    "*awsdns*",
    "*.googledomains.com",
    "ns-cloud-*.googledomains.com",
    "*.azure-dns.*",
    "*.nsone.net",
    "*.dynect.net",
    "*.ultradns.*",
    "*.ovh.net",
    "*.home.pl",
    "*.nazwa.pl",
    "*.hostinger.*",
    "*.dns-parking.com",
    "*.parkingcrew.net",
    "*.sedoparking.com",
    "*.bodis.com",
    "*.name-services.com",
    "*.dnsowl.com",
    "*.hostgator.com",
    "*.bluehost.com",
    "*.digitalocean.com",
    "*.linode.com",
    "*.vercel-dns.com",
    "*.netlify.com",
]


# Typowe pola formularzy (nie są śladem zestawu „kit”)
_DEFAULT_FORM_FIELDS = """
name firstname first_name fname lastname last_name lname surname fullname full_name email e-mail mail
phone tel telephone phone_number phonenumber mobile cell country country_code countrycode city address
zip zipcode postcode message comment comments submit send csrf _token token csrfmiddlewaretoken password
pass username login remember search q s lang language agree terms privacy consent checkbox captcha
g-recaptcha-response h-captcha-response recaptcha utm_source utm_medium utm_campaign utm_content utm_term
fbclid gclid id action redirect return_url url age
"""

# Zakresy adresów brzegowych Cloudflare (publiczna lista) — wspólne dla tysięcy niezależnych witryn
_CDN_NETWORKS = [
    "173.245.48.0/20",
    "103.21.244.0/22",
    "103.22.200.0/22",
    "103.31.4.0/22",
    "141.101.64.0/18",
    "108.162.192.0/18",
    "190.93.240.0/20",
    "188.114.96.0/20",
    "197.234.240.0/22",
    "198.41.128.0/17",
    "162.158.0.0/15",
    "104.16.0.0/13",
    "104.24.0.0/14",
    "172.64.0.0/13",
    "131.0.72.0/22",
    "2400:cb00::/32",
    "2606:4700::/32",
    "2803:f800::/32",
    "2405:b500::/32",
    "2405:8100::/32",
    "2a06:98c0::/29",
    "2c0f:f248::/32",
]

# Tytuły stron błędów, wyzwań anty-bot, parkingów i platform (po normalizacji `fold`)
_GENERIC_TITLE_RE = re.compile(
    r"^(?:"
    r"just a moment|attention required|access denied|forbidden|not found|error\b|bad gateway|"
    r"service unavailable|gateway time-?out|page not found|site not found|strona nie (?:zostala )?znaleziona|"
    r"nie znaleziono|blad\b|suspended|account (?:has been )?suspended|this account has been suspended|"
    r"domain (?:is )?for sale|domena (?:jest )?na sprzedaz|parked|this domain|index of\b|welcome to nginx|"
    r"apache2?\b.*default page|it works|default web site page|under construction|coming soon|"
    r"loading\b|redirect|moved permanently|object moved|document moved|this site can.?t be reached|ddos-guard|"
    r"security check|captcha|one more step|please wait|checking your browser|"
    r"(?:log ?in|sign ?in|zaloguj(?: sie)?)$|"
    r"(?:facebook|instagram|google|youtube|tiktok|telegram|whatsapp|bitly|tinyurl)\b|"
    r"[45]\d\d (?:forbidden|not found|internal|bad|service|gateway|error|unauthorized|too many)|"
    r"website expired|domain expired|strona w budowie|wkrotce"
    r")"
)

_TRACKER_PLACEHOLDER_RE = re.compile(r"x{4,}|^0+$|your|pixel_?id|^(ua|g|gtm|aw)-?$", re.I)


@dataclass(frozen=True)
class Stoplists:
    domains: frozenset[str] = frozenset()
    shorteners: frozenset[str] = frozenset()
    hosting: frozenset[str] = frozenset()
    asns: frozenset[str] = frozenset()  # same numery, np. "13335"
    nameserver_patterns: tuple[str, ...] = ()
    tls_issuers: tuple[str, ...] = ()
    form_fields: frozenset[str] = frozenset()
    titles: frozenset[str] = frozenset()
    registrars: frozenset[str] = frozenset()
    trackers: frozenset[str] = frozenset()
    favicons: frozenset[str] = frozenset()
    ips: frozenset[str] = frozenset()
    networks: tuple = field(default_factory=tuple)  # ipaddress.ip_network


def _words(block: str) -> set[str]:
    return {w.strip().lower() for w in block.split() if w.strip()}


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(k) for k in value]
    if isinstance(value, Iterable):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _asn_number(value: str) -> str | None:
    m = re.search(r"(?i)\bAS\s*(\d+)\b", value) or re.match(r"^\s*(\d+)\b", value)
    return m.group(1) if m else None


def _stoplist_path() -> Path:
    from slad.settings import CONFIG_DIR

    return CONFIG_DIR / "stoplists.yaml"


@functools.lru_cache(maxsize=8)
def _load_stoplists_cached(path_str: str, mtime_ns: int) -> Stoplists:
    data: dict = {}
    path = Path(path_str)
    if mtime_ns >= 0:
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
            elif loaded is not None:
                log.warning("Plik %s nie zawiera słownika list — pomijam", path)
        except Exception as e:  # noqa: BLE001 — zła lista nie może zatrzymać analizy
            log.warning("Nie udało się wczytać %s: %s", path, e)

    def items(key: str) -> list[str]:
        return [s.strip() for s in _as_list(data.get(key)) if s and s.strip()]

    domains = _words(_DEFAULT_STOP_DOMAINS) | {norm_host(d) for d in items("domains")}
    shorteners = _words(_SHORTENERS) | {norm_host(d) for d in items("shorteners")}
    hosting = _words(_HOSTING_DOMAINS) | {norm_host(d) for d in items("hosting_domains") + items("hosting")}
    asns = set(_DEFAULT_STOP_ASNS.split()) | {n for n in (_asn_number(a) for a in items("asns")) if n}
    ns_patterns = tuple(dict.fromkeys(_DEFAULT_NS_PATTERNS + [p.lower() for p in items("nameserver_patterns")]))
    # wystawca certyfikatu nigdy nie jest przesłanką (przesłanką jest odcisk / wspólne SAN) — lista tylko informacyjna
    tls_issuers = tuple(dict.fromkeys(fold(p) for p in items("tls_issuers")))
    form_fields = _words(_DEFAULT_FORM_FIELDS) | {f.lower() for f in items("form_fields")}
    networks = []
    ips = set()
    for net in _CDN_NETWORKS + items("ip_networks") + items("networks"):
        try:
            networks.append(ipaddress.ip_network(net, strict=False))
        except ValueError:
            log.debug("Niepoprawna sieć na liście: %s", net)
    for ip in items("ips"):
        if "/" in ip:
            try:
                networks.append(ipaddress.ip_network(ip, strict=False))
            except ValueError:
                pass
        else:
            ips.add(ip.lower())
    return Stoplists(
        domains=frozenset(d for d in domains if d),
        shorteners=frozenset(d for d in shorteners if d),
        hosting=frozenset(d for d in hosting if d),
        asns=frozenset(asns),
        nameserver_patterns=ns_patterns,
        tls_issuers=tuple(t for t in tls_issuers if t),
        form_fields=frozenset(form_fields),
        titles=frozenset(fold(t) for t in items("titles")),
        registrars=frozenset(fold(r) for r in items("registrars")),
        trackers=frozenset(t.lower() for t in items("trackers") + items("tracker_ids")),
        favicons=frozenset(items("favicons") + items("favicon_mmh3")),
        ips=frozenset(ips),
        networks=tuple(networks),
    )


def load_stoplists(path: str | Path | None = None) -> Stoplists:
    """Listy wykluczeń: wbudowane + config/stoplists.yaml (plik opcjonalny; przeładowywany po zmianie)."""
    p = Path(path) if path else _stoplist_path()
    try:
        mtime = p.stat().st_mtime_ns
    except OSError:
        mtime = -1
    return _load_stoplists_cached(str(p), mtime)


# ---------------------------------------------------------------- normalizacja


_PL_FOLD = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def fold(text: str | None) -> str:
    """Małe litery, bez znaków diakrytycznych (ł → l), pojedyncze spacje — klucz porównań."""
    if not text:
        return ""
    s = str(text).translate(_PL_FOLD)
    if not s.isascii():
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.casefold().replace("ł", "l").split())


def norm_host(value: str | None) -> str:
    """Host z adresu / nazwy: małe litery, bez www., portu, kropki końcowej; IDN → punycode."""
    if not value:
        return ""
    h = str(value).strip().lower()
    if "://" in h:
        try:
            h = urlsplit(h).hostname or ""
        except ValueError:
            return ""
    else:
        h = h.split("/")[0].split("?")[0].split("#")[0]
        h = h.rsplit("@", 1)[-1]
        if h.count(":") == 1:
            h = h.split(":")[0]
    h = h.strip().strip(".").strip("[]")
    if h.startswith("*."):
        h = h[2:]
    while h.startswith("www.") or h.startswith("ww."):
        h = h.split(".", 1)[1]
    if h and not h.isascii():
        try:
            h = h.encode("idna").decode("ascii")
        except UnicodeError:
            pass
    return h


@functools.lru_cache(maxsize=1)
def _tld_extractor():
    try:
        import tldextract

        return tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)
    except Exception as e:  # noqa: BLE001
        log.warning("tldextract niedostępny (%s) — domena = dwie ostatnie etykiety", e)
        return None


def is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


@functools.lru_cache(maxsize=65536)
def registered_domain(host: str) -> str:
    """eTLD+1 (z prywatnymi sufiksami PSL, np. foo.netlify.app); nieznany TLD (np. .test) → dwie ostatnie etykiety."""
    h = norm_host(host)
    if not h or "." not in h or is_ip(h):
        return ""
    ex = _tld_extractor()
    if ex is not None:
        try:
            r = ex(h)
            dom = getattr(r, "top_domain_under_public_suffix", None)
            if dom is None:
                dom = r.registered_domain
            if dom:
                return dom
        except Exception:  # noqa: BLE001
            pass
    labels = [x for x in h.split(".") if x]
    return ".".join(labels[-2:]) if len(labels) >= 2 else ""


def site_domain(host: str, stop: Stoplists | None = None) -> str:
    """Tożsamość witryny: domena zarejestrowana, a na współdzielonym hostingu — pełny host."""
    stop = stop or load_stoplists()
    h = norm_host(host)
    reg = registered_domain(h)
    if not reg:
        return ""
    if reg in stop.hosting and h != reg:
        return h
    return reg


def _domain_stopped(dom: str, stop: Stoplists) -> bool:
    if not dom:
        return True
    if dom in stop.domains or dom in stop.shorteners or dom in stop.hosting:
        return True
    reg = registered_domain(dom)
    return bool(reg) and (reg in stop.domains or reg in stop.shorteners)


def norm_phone(value: str | None) -> str | None:
    if not value:
        return None
    digits = re.sub(r"\D", "", str(value))
    while digits.startswith("00"):
        digits = digits[2:]
    if len(digits) == 9:  # numer krajowy (PL) bez kierunkowego
        digits = "48" + digits
    if len(digits) < 8 or len(digits) > 15:
        return None
    core = digits[-9:]
    if len(set(core)) <= 2 or core in {"123456789", "987654321", "012345678"}:
        return None
    return "+" + digits


def norm_email(value: str | None) -> str | None:
    if not value:
        return None
    v = str(value).strip().strip("<>").lower()
    v = v.removeprefix("mailto:")
    if v.count("@") != 1:
        return None
    local, dom = v.split("@")
    if not local or "." not in dom or dom in {"example.com", "example.org", "domain.com", "email.com", "test.com"}:
        return None
    return v


def norm_telegram(value: str | None) -> str | None:
    if not value:
        return None
    v = str(value).strip().lower()
    v = re.sub(r"^(https?://)?(www\.)?(t\.me|telegram\.me|telegram\.dog)/", "", v)
    v = v.lstrip("@").split("?")[0].strip("/")
    v = v.removeprefix("s/")
    if len(v) < 4 or v in {"joinchat", "share", "addstickers", "proxy", "iv", "telegram"}:
        return None
    return v


def norm_iban(value: str | None) -> str | None:
    if not value:
        return None
    v = re.sub(r"[\s-]", "", str(value)).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", v):
        return None
    return v


def norm_wallet(value: str | None) -> str | None:
    if not value:
        return None
    v = str(value).strip()
    if len(v) < 20:
        return None
    if v.lower().startswith(("0x", "bc1", "tb1", "ltc1")):
        v = v.lower()
    return v


def norm_tracker(value: str | None, stop: Stoplists) -> str | None:
    if not value:
        return None
    v = str(value).strip().lower().replace(" ", "")
    ident = v.split(":", 1)[-1]
    if len(ident) < 4 or _TRACKER_PLACEHOLDER_RE.search(ident) or v in stop.trackers or ident in stop.trackers:
        return None
    return v


def norm_title(value: str | None, stop: Stoplists) -> str | None:
    if not value:
        return None
    t = fold(value).strip(" -|·:•—–")
    if len(t) < 12 or is_generic_title(t) or t in stop.titles:
        return None
    return t


def is_generic_title(title: str | None) -> bool:
    t = fold(title)
    return bool(t) and bool(_GENERIC_TITLE_RE.match(t))


_LOCAL_NETWORKS = tuple(
    ipaddress.ip_network(n)
    for n in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::/128",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
        "ff00::/8",
    )
)


def _ip_ok(value: str, stop: Stoplists) -> str | None:
    """Adres IP jako wskaźnik: bez adresów lokalnych / prywatnych / sinkhole i bez adresów brzegowych CDN.

    Zakresy dokumentacyjne (TEST-NET) są dopuszczone — używa ich zbiór syntetyczny; w realnych danych nie występują.
    """
    try:
        ip = ipaddress.ip_address(str(value).strip())
    except ValueError:
        return None
    if any(ip.version == n.version and ip in n for n in _LOCAL_NETWORKS):
        return None
    s = str(ip)
    if s in stop.ips or any(ip.version == n.version and ip in n for n in stop.networks):
        return None
    return s


def _ns_stopped(ns: str, stop: Stoplists) -> bool:
    return _ns_stopped_cached(ns, stop.nameserver_patterns)


@functools.lru_cache(maxsize=65536)
def _ns_stopped_cached(ns: str, patterns: tuple[str, ...]) -> bool:
    for p in patterns:
        if p in ns or fnmatch(ns, p):
            return True
        if any(ch in p for ch in "\\^$+()|{"):
            try:
                if re.search(p, ns):
                    return True
            except re.error:
                pass
    return False


def _shared_cdn_cert(sans: Iterable[str]) -> bool:
    """Współdzielony certyfikat brzegowy CDN (np. dawne sni.cloudflaressl.com) — SAN-y obcych witryn."""
    return any("cloudflaressl.com" in str(s).lower() for s in sans)


def registrar_is_common(registrar: str | None, stop: Stoplists | None = None) -> bool:
    """Popularny rejestrator (lista `registrars`, dopasowanie po podciągu) — zbieżność dat jest słabsza."""
    stop = stop or load_stoplists()
    f = fold(registrar)
    return bool(f) and any(r and r in f for r in stop.registrars)


def _url_parts(url: str) -> tuple[str, str]:
    """(host, ścieżka) z adresu; dopuszcza adresy bez schematu."""
    u = str(url).strip()
    if not u:
        return "", ""
    if "://" not in u:
        u = "http://" + u.lstrip("/")
    try:
        sp = urlsplit(u)
        return norm_host(sp.hostname or ""), sp.path or ""
    except ValueError:
        return "", ""


def person_key(name: str | None) -> str:
    """Klucz osoby: „Marek_Zawadzki” (katalog bazy twarzy) i „Marek Zawadzki” (tekst) → „marek zawadzki”."""
    return fold(str(name or "").replace("_", " "))


# ---------------------------------------------------------------- strona „generyczna”


def final_domain(fs: FeatureSet) -> str:
    """Tożsamość witryny końcowego adresu łańcucha (site_domain) albo ""."""
    inf = fs.infra
    url = inf.final_url or (inf.redirect_chain[-1] if inf.redirect_chain else None)
    if not url:
        return ""
    host, _ = _url_parts(url)
    return site_domain(host) if host and not is_ip(host) else ""


def chain_roles(fs: FeatureSet, stop: Stoplists | None = None) -> tuple[str, set[str]]:
    """(domena docelowa, domeny pośrednie) łańcucha wejście → przekierowania → adres końcowy.

    Domena docelowa = ostatnia w łańcuchu domena spoza list wykluczeń (gdy cloaker odsyła na google.com,
    domeną docelową pozostaje domena oszustwa). Bez adresu końcowego (martwa strona) nic nie jest „pośrednie”.
    """
    stop = stop or load_stoplists()
    inf = fs.infra
    seq = [u for u in inf.input_urls if u] + [u for u in inf.redirect_chain if u]
    if inf.final_url:
        seq.append(inf.final_url)
    doms: list[str] = []
    for u in seq:
        host, _ = _url_parts(u)
        if not host or is_ip(host):
            continue
        d = site_domain(host, stop)
        if d and not _domain_stopped(d, stop):
            doms.append(d)
    if not doms:
        return "", set()
    landing = doms[-1]
    if not inf.final_url and len(set(doms)) == 1:
        return landing, set()
    return landing, {d for d in doms if d != landing}


def page_on_platform(fs: FeatureSet, stop: Stoplists | None = None) -> bool:
    """Łańcuch kończy się na platformie (facebook.com, parking domen…) — cechy strony opisują platformę."""
    stop = stop or load_stoplists()
    dom = final_domain(fs)
    return bool(dom) and _domain_stopped(dom, stop)


def page_is_generic(fs: FeatureSet, stop: Stoplists | None = None) -> bool:
    """Strona docelowa to platforma, błąd HTTP, parking albo wyzwanie anty-bot — jej cechy nie są śladem kampanii."""
    stop = stop or load_stoplists()
    inf = fs.infra
    if page_on_platform(fs, stop):
        return True
    if inf.http_status is not None and inf.http_status >= 400:
        return True
    return is_generic_title(inf.title) or (bool(inf.title) and fold(inf.title) in stop.titles)


# ---------------------------------------------------------------- wskaźniki


def indicators(fs: FeatureSet, stop: Stoplists | None = None) -> dict[str, set[str]]:
    """Dyskretne wskaźniki materiału (znormalizowane, bez wartości z list wykluczeń). Puste rodzaje są pomijane."""
    stop = stop or load_stoplists()
    out: dict[str, set[str]] = {}

    def put(kind: str, value: str | None) -> None:
        if value:
            out.setdefault(kind, set()).add(value)

    inf = fs.infra
    txt = fs.text
    iocs = txt.iocs or {}
    on_platform = page_on_platform(fs, stop)
    generic = on_platform or page_is_generic(fs, stop)

    # --- adresy: domeny (docelowe / pośrednie), hosty, skrócone linki
    _, intermediate = chain_roles(fs, stop)
    urls: list[str] = []
    urls += [u for u in inf.input_urls if u]
    urls += [u for u in inf.redirect_chain if u]
    if inf.final_url:
        urls.append(inf.final_url)
    text_urls = [u for u in iocs.get("url", []) if u]
    hosts: set[str] = set()
    for u in urls + text_urls:
        host, path = _url_parts(u)
        if not host:
            continue
        hosts.add(host)
        reg = registered_domain(host)
        if (host in stop.shorteners or reg in stop.shorteners) and len(path.strip("/")) >= 3:
            raw_path = urlsplit(u if "://" in u else "http://" + u).path.rstrip("/")
            put("short_url", f"{host}{raw_path}")
    hosts |= {norm_host(h) for h in inf.hosts if h}
    for h in iocs.get("domain", []):
        hosts.add(norm_host(h))
    for d in inf.domains:
        d = norm_host(d)
        if d and d not in stop.hosting:
            hosts.add(d)
    for h in hosts:
        if not h or is_ip(h):
            continue
        dom = site_domain(h, stop)
        if not dom or _domain_stopped(dom, stop):
            continue
        if dom in intermediate:
            put("redirect_domain", dom)
            continue
        put("domain", dom)
        if h != dom and not h.startswith(("www.", "m.")):
            put("host", h)

    # --- strona docelowa (pomijane, gdy to platforma / błąd / parking / wyzwanie anty-bot)
    if not on_platform:
        for t in inf.tracker_ids:
            v = norm_tracker(t, stop)
            if not v or v.startswith("tracker:"):  # rodzaj oprogramowania trackera kliknięć, nie operator
                continue
            if v.startswith("form:"):  # identyfikator oferty / strumienia z ukrytego pola formularza
                put("form_offer", v.split(":", 1)[1])
            else:
                put("tracker_id", v)
        for d in inf.script_domains:
            dom = site_domain(d, stop)
            if dom and not _domain_stopped(dom, stop):
                put("script_domain", dom)
        asn_numbers = [n for n in (_asn_number(str(a)) for a in inf.asns) if n]
        cdn_only = bool(asn_numbers) and all(n in stop.asns for n in asn_numbers)
        if not cdn_only:  # adresy brzegowe CDN są wspólne dla tysięcy niezależnych witryn
            for ip in inf.ips:
                put("ip", _ip_ok(ip, stop))
        for a in inf.asns:
            n = _asn_number(str(a))
            if n and n not in stop.asns:
                put("asn", "as" + n)
        for ns in inf.nameservers:
            ns_n = str(ns).strip().lower().rstrip(".")
            if ns_n and not _ns_stopped(ns_n, stop):
                put("nameserver", ns_n)
        if not _shared_cdn_cert(inf.tls_sans):
            if inf.tls_fingerprint:
                put("tls_fingerprint", re.sub(r"[^0-9a-f]", "", inf.tls_fingerprint.lower()) or None)
            for san in list(inf.tls_sans) + list(inf.ct_domains):
                dom = site_domain(san, stop)
                if dom and not _domain_stopped(dom, stop):
                    put("cert_domain", dom)
    # cechy samej strony (tytuł, ikona, formularz) — tylko gdy strona nie jest generyczna
    if not generic:
        put("title", norm_title(inf.title, stop))
        if inf.favicon_mmh3 is not None:
            fav = str(inf.favicon_mmh3)
            if fav not in stop.favicons and fav != "0":
                put("favicon", fav)
        for fa in inf.form_actions:
            host, path = _url_parts(fa) if ("://" in str(fa) or str(fa).startswith("//")) else ("", "")
            dom = site_domain(host, stop) if host else ""
            if dom and not _domain_stopped(dom, stop):
                put("form_action", f"{dom}{path.lower().rstrip('/')}")
        fields = sorted({str(f).strip().lower() for f in inf.form_fields if f} - stop.form_fields)
        fields = [f for f in fields if 2 <= len(f) <= 40]
        if len(fields) >= 2:
            put("form_fields", ",".join(fields[:12]))

    # --- identyfikatory kontaktowe i płatnicze z treści
    for key, values in iocs.items():
        if key.startswith("crypto"):
            for v in values:
                put("crypto_wallet", norm_wallet(v))
    for v in iocs.get("phone", []):
        put("phone", norm_phone(v))
    for v in iocs.get("email", []):
        put("email", norm_email(v))
    for v in iocs.get("telegram", []):
        put("telegram", norm_telegram(v))
    for v in iocs.get("whatsapp", []):
        put("whatsapp", norm_phone(re.sub(r"^(https?://)?(api\.)?(wa\.me|whatsapp\.com)/(send\?phone=)?", "", str(v))))
    for v in iocs.get("iban", []):
        put("iban", norm_iban(v))

    # --- osoby, marki, narracja, słowa kluczowe
    for p in txt.persons:
        k = person_key(p)
        if len(k) >= 4:
            put("person", k)
    for img in fs.images:
        for face in img.faces:
            if face.identity:
                k = person_key(face.identity)
                if len(k) >= 4:
                    put("person", k)
    for b in txt.brands:
        k = fold(b)
        if len(k) >= 2:
            put("brand", k)
    for n in txt.narratives:
        k = fold(str(n).replace("_", " "))
        if k:
            put("narrative", k)
    for kw in txt.keywords:
        k = fold(kw)
        if len(k) >= 3 and not k.isdigit():
            put("keyword", k)
    return out


# ---------------------------------------------------------------- prezentacja wartości

_TRACKER_NAMES = {
    "ga": "Google Analytics",
    "ua": "Google Analytics",
    "gtag": "Google Analytics 4",
    "ga4": "Google Analytics 4",
    "gtm": "Google Tag Manager",
    "aw": "Google Ads",
    "gads": "Google Ads",
    "fbpixel": "Facebook Pixel",
    "fb": "Facebook Pixel",
    "pixel": "Facebook Pixel",
    "ym": "Yandex Metrica",
    "yandex": "Yandex Metrica",
    "tiktok": "TikTok Pixel",
    "keitaro": "Keitaro",
    "binom": "Binom",
    "hotjar": "Hotjar",
    "clarity": "Microsoft Clarity",
}


def pretty_tracker(value: str) -> str:
    """„fbpixel:123” → „Facebook Pixel 123”, „gtm:gtm-ab12” → „Google Tag Manager GTM-AB12”."""
    if ":" not in value:
        return value
    prefix, ident = value.split(":", 1)
    name = _TRACKER_NAMES.get(prefix, prefix)
    if prefix in {"ga", "ua", "gtag", "ga4", "gtm", "aw", "gads", "tiktok", "clarity"}:
        ident = ident.upper()
    return f"{name} {ident}"


@functools.lru_cache(maxsize=4)
def _narrative_labels(path_str: str, mtime_ns: int) -> dict[str, str]:
    try:
        data = yaml.safe_load(Path(path_str).read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    if isinstance(data, dict):
        for tag, spec in data.items():
            if isinstance(spec, dict) and spec.get("label_pl"):
                out[fold(str(tag).replace("_", " "))] = str(spec["label_pl"])
    return out


def narrative_label(tag: str, short: bool = True) -> str:
    """Opis narracji z config/narratives.yaml (label_pl); short=True — część przed „ / ” lub „ (”."""
    from slad.settings import CONFIG_DIR

    path = CONFIG_DIR / "narratives.yaml"
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = -1
    label = _narrative_labels(str(path), mtime).get(fold(str(tag).replace("_", " ")))
    if not label:
        return str(tag).replace("_", " ")
    if short:
        label = re.split(r"\s+\(", label)[0].strip()
        first = re.split(r"\s+/\s+", label)[0].strip()
        if len(first.split()) >= 2:  # „Fałszywy wywiad / artykuł…” → „Fałszywy wywiad”; „Bot / …” zostaje w całości
            label = first
    return label


def _describe_tracker(original: str) -> str:
    try:
        from slad.features.infra.html import describe_tracker

        return describe_tracker(original)
    except Exception:  # noqa: BLE001 — moduł infrastruktury opcjonalny
        return pretty_tracker(original.lower())


def display_value(kind: str, value: str, *features: FeatureSet) -> str:
    """Czytelna postać wartości wskaźnika (oryginalna pisownia osoby / marki, nazwa trackera, numer ASN)."""
    if kind == "tracker_id":
        for fs in features:
            if fs is None:
                continue
            for t in fs.infra.tracker_ids:
                if str(t).strip().lower().replace(" ", "") == value:
                    return _describe_tracker(str(t).strip())
        return pretty_tracker(value)
    if kind == "telegram":
        return value if value.startswith(("+", "joinchat")) else "@" + value
    if kind in {"person", "brand", "narrative", "asn"}:
        for fs in features:
            if fs is None:
                continue
            if kind == "person":
                cands = list(fs.text.persons) + [f.identity for img in fs.images for f in img.faces if f.identity]
                for c in cands:
                    if person_key(c) == value:
                        return str(c).replace("_", " ")
            elif kind == "brand":
                for c in fs.text.brands:
                    if fold(c) == value:
                        return c
            elif kind == "narrative":
                for c in fs.text.narratives:
                    if fold(str(c).replace("_", " ")) == value:
                        return narrative_label(str(c))
            elif kind == "asn":
                for c in fs.infra.asns:
                    n = _asn_number(str(c))
                    if n and "as" + n == value:
                        return str(c)
        if kind == "person":
            return value.title()
        if kind == "asn":
            return value.upper()
    return value
