"""infra_features: meta.infra + zapisany HTML (offline), kroki sieciowe na atrapach, pamięć podręczna i tryb offline,
strony „ogólne”, platformy, opakowania przekierowań, MHTML, złe dane wejściowe, integracja z extract()."""

from __future__ import annotations

import base64
import socket
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import dns.resolver
import httpx
import pytest

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.features import ExtractContext  # noqa: E402
from slad.features.infra import fetch as fetch_mod  # noqa: E402
from slad.features.infra import infra_features  # noqa: E402
from slad.features.infra import netinfo as netinfo_mod  # noqa: E402
from slad.features.infra import tls as tls_mod  # noqa: E402
from slad.features.infra.cache import JsonCache  # noqa: E402
from slad.features.infra.favicon import favicon_hash  # noqa: E402
from slad.features.infra.fetch import FetchResult  # noqa: E402
from slad.models import InfraFeatures, Material, SourceKind  # noqa: E402
from slad.settings import NetworkSettings, Settings  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "infra"
LANDING = (FIX / "landing_a1.html").read_text(encoding="utf-8")
PARKED = (FIX / "parked.html").read_text(encoding="utf-8")
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
FINAL = "https://www.scam-news.pl/lp"


class NetworkUsed(BaseException):  # noqa: N818 — BaseException: nie da się go „połknąć” przez except Exception
    pass


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    """Twarda blokada prawdziwej sieci (HTTP, gniazda, DNS) we wszystkich testach tego pliku."""

    def deny(*args, **kwargs):
        raise NetworkUsed("prawdziwa sieć w teście!")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(dns.resolver.Resolver, "resolve", deny)
    monkeypatch.setattr(fetch_mod, "resolve_addresses", lambda host: ["93.184.216.34"])  # bez prawdziwego DNS


class FakeNet:
    """Atrapy funkcji sieciowych modułu infrastruktury (podstawiane w modułach, z dziennikiem wywołań)."""

    def __init__(self) -> None:
        self.calls: dict[str, list] = defaultdict(list)
        self.page_status = 200
        self.page_html: str | None = LANDING
        self.fetch_error: str | None = None
        self.disabled = False

    def _log(self, name: str, arg) -> None:
        if self.disabled:
            raise NetworkUsed(f"{name} wywołane mimo pamięci podręcznej / trybu offline")
        self.calls[name].append(arg)

    def fetch_chain(self, url, cfg, client=None, **kw):
        self._log("fetch", url)
        if self.fetch_error:
            return FetchResult(chain=[url], final_url=url, status=None, error=self.fetch_error)
        pages = [(FINAL, self.page_html)] if self.page_html is not None else []
        return FetchResult(
            chain=[url, FINAL],
            final_url=FINAL,
            status=self.page_status,
            headers={"server": "nginx", "content-type": "text/html"},
            body=self.page_html,
            content_type="text/html",
            pages=pages,
        )

    def fetch_bytes(self, url, cfg, client=None, **kw):
        self._log("favicon", url)
        if url.endswith("/static/img/fav.png"):
            return PNG, "image/png", None
        return None, None, "HTTP 404"

    def dns_info(self, host, timeout=5.0, **kw):
        self._log("dns", host)
        if self.fetch_error:
            return {"a": [], "aaaa": [], "cname": [], "ns": [], "mx": [], "errors": ["dns: domena nie istnieje"]}
        return {
            "a": ["104.21.5.6"],
            "aaaa": [],
            "cname": [],
            "ns": ["kate.ns.cloudflare.com"],
            "mx": ["mx.scam-news.pl"],
            "errors": [],
        }

    def asn_lookup(self, ip, timeout=5.0, **kw):
        self._log("asn", ip)
        return "AS13335 CLOUDFLARENET", None

    def rdap_ip(self, ip, timeout=8.0, **kw):
        self._log("rdap_ip", ip)
        return {"name": None, "error": "rdap: brak"}

    def rdap_domain(self, domain, timeout=8.0, **kw):
        self._log("rdap", domain)
        return {
            "registrar": "NameSilo, LLC",
            "created": "2025-10-01T00:00:00+00:00",
            "expires": None,
            "updated": None,
            "nameservers": ["ns1.dnsowl.com"],
            "status": ["active"],
            "error": None,
        }

    def tls_info(self, host, port=443, timeout=8.0):
        self._log("tls", host)
        return {
            "issuer": "Let's Encrypt (R11)",
            "subject": host,
            "fingerprint": "ab" * 32,
            "sans": ["scam-news.pl", "www.scam-news.pl"],
            "not_before": None,
            "not_after": None,
            "error": None,
        }

    def crtsh_lookup(self, domain, timeout=8.0, **kw):
        self._log("crtsh", domain)
        return ["www.scam-news.pl", "panel.scam-news.pl", "inna-domena.pl"], None


@pytest.fixture
def net(monkeypatch) -> FakeNet:
    fake = FakeNet()
    monkeypatch.setattr(fetch_mod, "fetch_chain", fake.fetch_chain)
    monkeypatch.setattr(fetch_mod, "fetch_bytes", fake.fetch_bytes)
    monkeypatch.setattr(netinfo_mod, "dns_info", fake.dns_info)
    monkeypatch.setattr(netinfo_mod, "asn_lookup", fake.asn_lookup)
    monkeypatch.setattr(netinfo_mod, "rdap_ip", fake.rdap_ip)
    monkeypatch.setattr(netinfo_mod, "rdap_domain", fake.rdap_domain)
    monkeypatch.setattr(tls_mod, "tls_info", fake.tls_info)
    monkeypatch.setattr(tls_mod, "crtsh_lookup", fake.crtsh_lookup)
    return fake


def make_ctx(cache_root: Path, enabled: bool, **net_kw) -> ExtractContext:
    return ExtractContext(settings=Settings(network=NetworkSettings(enabled=enabled, **net_kw)), cache_root=cache_root)


# ---------------------------------------------------------------- offline: meta.infra + zapisany HTML


def test_offline_seed_and_local_html(tmp_path):
    ctx = make_ctx(tmp_path / "cache", enabled=False)
    landing = "https://www.wiadomosci24-info.pl/art?pixel=123456789012345&fbclid=abc"
    m = Material(
        id="m1",
        source=SourceKind.RECORD,
        url="https://l.facebook.com/l.php?u=" + quote(landing, safe="") + "&h=AT0",
        html_path=str(FIX / "landing_a1.html"),
        meta={
            "infra": {
                "ips": ["104.21.1.1", "zły-ip"],
                "asn": "13335 | CLOUDFLARENET",
                "registrar": "NameSilo, LLC",
                "creation_date": "01.10.2025",
                "favicon_hash": "-123456",
            },
            "landing_domain": "wiadomosci24-info.pl",
            "image_url": "https://scontent.xx.fbcdn.net/obraz.jpg",  # pole obrazu — to nie adres strony
            "opis": "Zobacz http://ignored.example teraz",
        },
    )
    inf, text = infra_features(m, ctx)
    assert isinstance(inf, InfraFeatures)
    assert inf.input_urls == [
        "https://l.facebook.com/l.php?u=" + quote(landing, safe="") + "&h=AT0",
        "http://wiadomosci24-info.pl/",
    ]
    # opakowanie przekierowania Facebooka zdradza stronę docelową także bez sieci
    assert inf.redirect_chain[-1] == "https://www.wiadomosci24-info.pl/art?pixel=123456789012345"
    assert "fbpixel:123456789012345" in inf.tracker_ids  # piksel z parametru adresu
    assert {"gtag:G-ABCD123456", "gtm:GTM-AB12CD3", "ym:98765432", "form:offer_id=OF-7781"} <= set(inf.tracker_ids)
    assert inf.title == "Sensacyjne wyznanie: Jan Kowalski zarabia 5000 zł dziennie"
    assert inf.dom_simhash and inf.html_sha256
    # adres bazowy zapisanej strony = strona docelowa (nie l.facebook.com)
    assert "https://www.wiadomosci24-info.pl/api/register" in inf.form_actions
    assert {"facebook.com", "wiadomosci24-info.pl", "leadhub-api.com"} <= set(inf.domains)
    assert "fbcdn.net" not in inf.domains and "ignored.example" not in inf.domains
    assert inf.ips == ["104.21.1.1"]
    assert inf.asns == ["AS13335 CLOUDFLARENET"]
    assert inf.registrar == "NameSilo, LLC"
    assert inf.domain_created == datetime(2025, 10, 1, tzinfo=UTC)
    assert inf.favicon_mmh3 == -123456  # meta.infra ma pierwszeństwo przed stroną
    assert any(e.startswith("tryb offline") for e in inf.errors)
    assert inf.fetched_at is None and inf.from_cache is False
    assert "Jan Kowalski ujawnia" in text
    assert "tel:+48600111222" in text and "https://t.me/zarabiaj_z_janem" in text  # linki kontaktowe dla IOC


def test_meta_infra_formats(tmp_path):
    ctx = make_ctx(tmp_path, enabled=False)
    seed = {
        "Redirects": "https://a.example/x -> https://b.example/y → https://c.example/z?utm_source=fb",
        "Final": "https://c.example/z",
        "Status": "200",
        "trackers": ["UA-1234-1", "GTM-ABCD12", "gtag:G-ABCDEF1234", ""],
        "fb_pixel": 123456789012345,
        "IP": "104.21.5.6, 999.1.1.1",
        "ASN": [13335, "AS15169 GOOGLE"],
        "ns": "NS1.Example.COM.; ns2.example.com",
        "mx": "10 mx.example.com",
        "created": 1727740800,
        "favicon": "-987654321",
        "tls_fingerprint": "AB:CD:EF",
        "title": "  Tytuł   z  meta  ",
        "page_text": "Tekst strony z danych",
        "form_fields": "Phone, Email",
        "domains": ["WWW.Example.org"],
    }
    inf, text = infra_features(Material(id="s", source=SourceKind.RECORD, meta={"infra": seed}), ctx)
    assert inf.redirect_chain == ["https://a.example/x", "https://b.example/y", "https://c.example/z"]
    assert inf.final_url == "https://c.example/z" and inf.http_status == 200
    assert {"ga:UA-1234-1", "gtm:GTM-ABCD12", "gtag:G-ABCDEF1234", "fbpixel:123456789012345"} <= set(inf.tracker_ids)
    assert inf.ips == ["104.21.5.6"]
    assert inf.asns == ["AS13335", "AS15169 GOOGLE"]
    assert inf.nameservers == ["ns1.example.com", "ns2.example.com"]
    assert inf.mx == ["mx.example.com"]
    assert inf.domain_created == datetime(2024, 10, 1, tzinfo=UTC)
    assert inf.favicon_mmh3 == -987654321
    assert inf.tls_fingerprint == "abcdef"
    assert inf.title == "Tytuł z meta"
    assert inf.form_fields == ["email", "phone"]
    assert inf.domains[0] == "example.org" and {"a.example", "b.example", "c.example"} <= set(inf.domains)
    assert text == "Tekst strony z danych"
    assert inf.errors == []  # domeny .example nie są odpytywane — brak błędów trybu offline


def test_junk_inputs_never_crash(tmp_path):
    ctx = make_ctx(tmp_path, enabled=False)
    cases = [
        Material(
            id="j1", source=SourceKind.URL, url="brak danych", html_path="/nie/ma/strona.html", meta={"infra": "{zły"}
        ),
        Material(id="j2", source=SourceKind.RECORD, meta={"infra": ["lista"], "url": 12345, "link": None}),
        Material(
            id="j3", source=SourceKind.RECORD, meta={"infra": {"ip": {"a": 1}, "created": "wczoraj", "asn": None}}
        ),
        Material(id="j4", source=SourceKind.TEXT, text="sam tekst"),
    ]
    results = [infra_features(m, ctx) for m in cases]
    inf1, _ = results[0]
    assert "html: brak pliku strona.html" in inf1.errors
    assert "meta.infra: nieprawidłowy format (oczekiwano JSON)" in inf1.errors
    assert inf1.input_urls == [] and inf1.domains == []
    assert "meta.infra: nieprawidłowy format (oczekiwano słownika)" in results[1][0].errors
    assert results[2][0].domain_created is None and results[2][0].ips == []
    assert results[3] == (InfraFeatures(), "")


def test_local_generic_page_gives_no_content_features(tmp_path):
    ctx = make_ctx(tmp_path, enabled=False)
    m = Material(id="park", source=SourceKind.HTML, html_path=str(FIX / "parked.html"))
    inf, text = infra_features(m, ctx)
    assert inf.title is None and inf.dom_simhash is None and inf.tracker_ids == []
    assert any("parking domeny" in e for e in inf.errors)
    assert text == ""


def test_mhtml_saved_page(tmp_path):
    mhtml = (
        "From: <Saved by Blink>\r\n"
        "Snapshot-Content-Location: https://saved-scam.pl/oferta\r\n"
        "Subject: Oferta\r\n"
        "MIME-Version: 1.0\r\n"
        'Content-Type: multipart/related; type="text/html"; boundary="----Boundary--abc----"\r\n'
        "\r\n"
        "------Boundary--abc----\r\n"
        "Content-Type: text/html\r\n"
        "Content-Transfer-Encoding: quoted-printable\r\n"
        "Content-Location: https://saved-scam.pl/oferta\r\n"
        "\r\n"
        "<html><head><title>Zapisana oferta</title></head><body><p>Zarabiaj =C5=BCycie</p>"
        '<form action=3D"/send.php"><input name=3D"phone"></form></body></html>\r\n'
        "------Boundary--abc------\r\n"
    )
    f = tmp_path / "strona.mhtml"
    f.write_bytes(mhtml.encode("ascii"))
    inf, text = infra_features(Material(id="mh", source=SourceKind.HTML, html_path=str(f)), make_ctx(tmp_path, False))
    assert inf.title == "Zapisana oferta"
    assert "https://saved-scam.pl/oferta" in inf.input_urls
    assert inf.form_actions == ["https://saved-scam.pl/send.php"]
    assert "saved-scam.pl" in inf.domains
    assert "Zarabiaj życie" in text


def test_ctx_without_cache_method(tmp_path):
    ctx = SimpleNamespace(settings=Settings(network=NetworkSettings(enabled=False)))
    inf, text = infra_features(Material(id="x", source=SourceKind.HTML, html_path=str(FIX / "landing_b.html")), ctx)
    assert inf.favicon_mmh3 is not None  # favicona data: — liczona bez sieci
    assert inf.tracker_ids[:2] == ["tiktok:C9ABCDEF1234567890XY", "clarity:k3x9q2w7ab"]
    assert "Let AI trade for you" in text


# ---------------------------------------------------------------- sieć (atrapy) + pamięć podręczna


def test_online_pipeline_cache_and_offline_reuse(tmp_path, net):
    cache_root = tmp_path / "cache"
    m = Material(id="n1", source=SourceKind.URL, url="https://bit.ly/abc123?fbclid=XYZ")
    inf, text = infra_features(m, make_ctx(cache_root, enabled=True))
    assert net.calls["fetch"] == ["https://bit.ly/abc123?fbclid=XYZ"]  # pobieramy adres jak podany
    assert inf.input_urls == ["https://bit.ly/abc123"]
    assert inf.redirect_chain == ["https://bit.ly/abc123", FINAL]
    assert inf.final_url == FINAL and inf.http_status == 200
    assert inf.title.startswith("Sensacyjne wyznanie")
    assert inf.hosts == ["bit.ly", "www.scam-news.pl", "crm.leadhub-api.com"]
    assert inf.domains == ["bit.ly", "scam-news.pl", "leadhub-api.com"]
    assert inf.favicon_mmh3 == favicon_hash(PNG)
    assert net.calls["favicon"] == ["https://www.scam-news.pl/static/img/fav.png"]
    assert inf.ips == ["104.21.5.6"] and inf.asns == ["AS13335 CLOUDFLARENET"]
    assert inf.nameservers == ["kate.ns.cloudflare.com", "ns1.dnsowl.com"]  # DNS + RDAP
    assert inf.mx == ["mx.scam-news.pl"]
    assert net.calls["rdap"] == ["scam-news.pl"]  # domena końcowa; bit.ly (lista wyjątków) pominięta
    assert inf.registrar == "NameSilo, LLC" and inf.domain_created == datetime(2025, 10, 1, tzinfo=UTC)
    assert inf.tls_issuer == "Let's Encrypt (R11)" and inf.tls_fingerprint == "ab" * 32
    assert inf.tls_sans == ["scam-news.pl", "www.scam-news.pl"]
    assert inf.ct_domains == []  # crt.sh domyślnie wyłączone
    assert inf.errors == []
    assert inf.from_cache is False and inf.fetched_at is not None
    assert "Jan Kowalski ujawnia" in text

    # drugie uruchomienie: wszystko z pamięci podręcznej, bez żadnego zapytania sieciowego
    net.disabled = True
    inf2, text2 = infra_features(m, make_ctx(cache_root, enabled=True))
    assert inf2.from_cache is True
    assert inf2.model_dump(exclude={"from_cache"}) == inf.model_dump(exclude={"from_cache"})
    assert text2 == text

    # tryb offline z przeterminowaną pamięcią podręczną: stare dane są lepsze niż żadne
    offline = make_ctx(cache_root, enabled=False, cache_ttl_h=1e-9)
    inf3, _ = infra_features(m, offline)
    assert inf3.from_cache is True
    assert inf3.model_dump(exclude={"from_cache"}) == inf.model_dump(exclude={"from_cache"})


def test_crtsh_when_enabled(tmp_path, net):
    m = Material(id="c", source=SourceKind.URL, url="https://www.scam-news.pl/lp")
    inf, _ = infra_features(m, make_ctx(tmp_path, enabled=True, crtsh=True))
    assert net.calls["crtsh"] == ["scam-news.pl"]
    assert inf.ct_domains == ["panel.scam-news.pl", "inna-domena.pl"]  # bez hosta strony końcowej


def test_feature_flags_respected(tmp_path, net):
    m = Material(id="f", source=SourceKind.URL, url="https://www.scam-news.pl/lp")
    cfg = {"dns": False, "rdap": False, "tls": False, "favicon": False}
    inf, _ = infra_features(m, make_ctx(tmp_path, enabled=True, **cfg))
    assert net.calls["fetch"] and not any(net.calls[k] for k in ("dns", "asn", "rdap", "tls", "favicon"))
    assert inf.title and inf.ips == [] and inf.registrar is None and inf.favicon_mmh3 is None


def test_generic_landing_page_skips_content_features(tmp_path, net):
    net.page_html = PARKED
    inf, text = infra_features(
        Material(id="g", source=SourceKind.URL, url="https://bit.ly/x"), make_ctx(tmp_path, True)
    )
    assert inf.final_url == FINAL and inf.http_status == 200
    assert inf.title is None and inf.dom_simhash is None and inf.html_sha256 is None
    assert "gtag:G-PARK000001" not in inf.tracker_ids  # tracker firmy parkingowej to nie ślad kampanii
    assert "strona docelowa: parking domeny — pominięto cechy treści strony" in inf.errors
    assert net.calls["favicon"] == []
    assert text == ""
    assert inf.registrar == "NameSilo, LLC"  # dane rejestracji domeny nadal są wartościowe


def test_http_error_status(tmp_path, net):
    net.page_status = 404
    net.page_html = "<html><head><title>Oferta</title></head><body>" + "<p>x</p>" * 20 + "</body></html>"
    inf, _ = infra_features(Material(id="e", source=SourceKind.URL, url="https://bit.ly/y"), make_ctx(tmp_path, True))
    assert inf.http_status == 404 and inf.title is None
    assert "strona docelowa: błąd HTTP 404 — pominięto cechy treści strony" in inf.errors


def test_dead_domain(tmp_path, net):
    net.fetch_error = "nie można rozwiązać nazwy domeny"
    m = Material(id="d", source=SourceKind.URL, url="https://dead-scam.pl/lp")
    inf, text = infra_features(m, make_ctx(tmp_path, True))
    assert inf.final_url == "https://dead-scam.pl/lp" and inf.http_status is None
    assert "pobieranie: nie można rozwiązać nazwy domeny" in inf.errors
    assert "dns: domena nie istnieje" in inf.errors
    assert net.calls["tls"] == [] and net.calls["favicon"] == []  # serwer nie odpowiedział
    assert net.calls["rdap"] == ["dead-scam.pl"]  # rejestracja martwej domeny nadal się liczy
    assert inf.registrar == "NameSilo, LLC"
    assert inf.domains == ["dead-scam.pl"] and text == ""


def test_failed_fetch_is_retried_after_negative_ttl(tmp_path, net, monkeypatch):
    net.fetch_error = "przekroczono czas"
    m = Material(id="r", source=SourceKind.URL, url="https://slow-scam.pl/")
    ctx = make_ctx(tmp_path, True)
    infra_features(m, ctx)
    infra_features(m, ctx)  # świeży wpis z błędem — z pamięci podręcznej
    assert len(net.calls["fetch"]) == 1
    monkeypatch.setattr(JsonCache, "age_h", lambda self, key: 3.0)  # wpis z błędem starszy niż 2 h
    net.fetch_error = None
    inf, _ = infra_features(m, ctx)
    assert len(net.calls["fetch"]) == 2
    assert inf.http_status == 200


def test_platform_url_not_fetched(tmp_path, net):
    m = Material(id="p", source=SourceKind.URL, url="https://www.facebook.com/ads/library/?id=123")
    inf, text = infra_features(m, make_ctx(tmp_path, True))
    assert net.calls["fetch"] == [] and net.calls["dns"] == []
    assert any(e.startswith("pominięto pobieranie: adres platformy") for e in inf.errors)
    assert inf.domains == ["facebook.com"] and text == ""


def test_link_shim_is_unwrapped_and_fetched(tmp_path, net):
    target = "https://scam-news.pl/lp?pixel=123456789012345&fbclid=1"
    shim = "https://l.facebook.com/l.php?u=" + quote(target, safe="") + "&h=AT1"
    inf, _ = infra_features(Material(id="s", source=SourceKind.URL, url=shim), make_ctx(tmp_path, True))
    assert net.calls["fetch"] == [target]
    assert inf.redirect_chain[0].startswith("https://l.facebook.com/l.php?u=")
    assert inf.redirect_chain[1] == "https://scam-news.pl/lp?pixel=123456789012345"
    assert inf.redirect_chain[-1] == FINAL
    assert "fbpixel:123456789012345" in inf.tracker_ids


def test_reserved_domains_are_not_queried(tmp_path, net):
    m = Material(id="t", source=SourceKind.URL, url="https://scam-site.example/lp", meta={"infra": {"ip": "8.8.8.8"}})
    inf, _ = infra_features(m, make_ctx(tmp_path, True))
    assert not any(net.calls.values())
    assert inf.errors == [] and inf.domains == ["scam-site.example"] and inf.ips == ["8.8.8.8"]


def test_local_html_wins_over_live_page(tmp_path, net):
    net.page_html = (FIX / "landing_b.html").read_text(encoding="utf-8")  # dziś pod adresem jest inna strona
    m = Material(id="l", source=SourceKind.RECORD, url=FINAL, html_path=str(FIX / "landing_a2.html"))
    inf, text = infra_features(m, make_ctx(tmp_path, True))
    assert inf.title.startswith("Szokujące: Anna Nowak")  # zapisana kopia ma pierwszeństwo
    assert "tiktok:C9ABCDEF1234567890XY" in inf.tracker_ids  # listy z obu źródeł
    assert "Anna Nowak" in text and "Let AI trade" not in text


# ---------------------------------------------------------------- integracja z orkiestratorem cech


def test_extract_orchestrator_uses_infra(tmp_path, monkeypatch):
    """Kontrakt z slad.features.extract: InfraFeatures trafia do FeatureSet, a page_text — do modułu tekstu."""
    import types

    from slad.features import extract
    from slad.models import TextFeatures

    received: dict[str, str] = {}

    def fake_text_features(material, ocr_texts, page_text, ctx):
        received["page_text"] = page_text
        return TextFeatures(page_text=page_text, text=page_text)

    # atrapa modułu tekstu (prawdziwy należy do innego zespołu i jest testowany osobno)
    monkeypatch.setitem(sys.modules, "slad.features.text", types.SimpleNamespace(text_features=fake_text_features))
    ctx = make_ctx(tmp_path, enabled=False)
    m = Material(
        id="x1", source=SourceKind.HTML, html_path=str(FIX / "landing_a1.html"), url="https://scam-news.example/a"
    )
    fs = extract(m, ctx)
    assert fs.infra.title.startswith("Sensacyjne wyznanie")
    assert "infrastruktura" in fs.available
    assert not any(e.startswith("infrastruktura:") for e in fs.errors)
    assert "Jan Kowalski ujawnia" in received["page_text"]
    assert "tel:+48600111222" in received["page_text"]


# ---------------------------------------------------------------- odporność: kolumny rekordu, stary cache, awarie


def test_top_level_record_columns_are_used(tmp_path):
    meta = {
        "IP": "104.21.5.6",
        "Registrar": "GoDaddy.com, LLC",
        "creation_date": "2025-11-02",
        "fb_pixel": "123456789012345",
        "title": "Tytuł REKLAMY — nie strony",  # nazwa ogólna: nie trafia do InfraFeatures.title
        "status": "aktywna",
    }
    inf, _ = infra_features(Material(id="t", source=SourceKind.RECORD, meta=meta), make_ctx(tmp_path, False))
    assert inf.ips == ["104.21.5.6"]
    assert inf.registrar == "GoDaddy.com, LLC"
    assert inf.domain_created == datetime(2025, 11, 2, tzinfo=UTC)
    assert "fbpixel:123456789012345" in inf.tracker_ids
    assert inf.title is None and inf.http_status is None


def test_old_cache_format_is_refetched_online(tmp_path, net):
    from slad.features.infra.domains import normalize_url

    ctx = make_ctx(tmp_path, True)
    url = "https://www.scam-news.pl/lp"
    ctx.cache("fetch").set(normalize_url(url), {"v": 0, "chain": [url], "status": 200})  # wpis w starym formacie
    inf, _ = infra_features(Material(id="o", source=SourceKind.URL, url=url), ctx)
    assert net.calls["fetch"] == [url]
    assert inf.title and inf.title.startswith("Sensacyjne wyznanie")


def test_platform_final_url_from_seed_is_not_queried(tmp_path, net):
    m = Material(id="pf", source=SourceKind.RECORD, meta={"infra": {"final_url": "https://www.facebook.com/strona"}})
    inf, _ = infra_features(m, make_ctx(tmp_path, True))
    assert not any(net.calls.values())
    assert inf.final_url == "https://www.facebook.com/strona" and inf.ips == []


def test_internal_error_returns_partial_result(tmp_path, monkeypatch):
    import slad.features.infra as infra_pkg

    def boom(candidates):
        raise RuntimeError("awaria")

    monkeypatch.setattr(infra_pkg, "_select_target", boom)
    m = Material(id="b", source=SourceKind.RECORD, url="https://scam.pl/", meta={"infra": {"registrar": "X"}})
    inf, text = infra_features(m, make_ctx(tmp_path, False))
    assert inf.registrar == "X" and inf.input_urls == ["https://scam.pl/"]
    assert "infrastruktura: błąd wewnętrzny (RuntimeError)" in inf.errors
    assert text == ""


def test_seed_errors_are_carried_over(tmp_path):
    seed = {
        "redirect_chain": ["https://dead.example/lp"],
        "final_url": None,
        "http_status": None,
        "errors": ["pobieranie: DNS: NXDOMAIN — domena nie istnieje", ""],
    }
    inf, _ = infra_features(Material(id="d", source=SourceKind.RECORD, meta={"infra": seed}), make_ctx(tmp_path, True))
    assert inf.errors == ["pobieranie: DNS: NXDOMAIN — domena nie istnieje"]
    assert inf.redirect_chain == ["https://dead.example/lp"] and inf.final_url is None
