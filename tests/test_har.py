"""Adapter HAR: łańcuch przekierowań, hosty/domeny, IP, skrypty, identyfikatory trackerów, TLS, błędy;
integracja z infrastrukturą w trybie offline. Dane syntetyczne (domeny .test, adresy TEST-NET), bez sieci."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import dns.resolver
import httpx
import pytest

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu
    sys.path.insert(0, _ROOT)

from slad.features import ExtractContext  # noqa: E402
from slad.features.infra import infra_features  # noqa: E402
from slad.ingest.har import har_to_seed, read_har  # noqa: E402
from slad.models import Material, SourceKind  # noqa: E402
from slad.settings import NetworkSettings, Settings  # noqa: E402

CHAIN = [
    "http://track.example.test/r?c=1",
    "https://landing.example.test/start",
    "https://www.offer.example.test/lp",
]


class NetworkUsed(BaseException):  # noqa: N818
    pass


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    def deny(*args, **kwargs):
        raise NetworkUsed("prawdziwa sieć w teście!")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(dns.resolver.Resolver, "resolve", deny)


def _entry(url, status=200, *, rtype="xhr", ip=None, redirect=None, mime="text/html", **extra):
    e = {
        "startedDateTime": "2026-11-21T10:00:00.000Z",
        "request": {"method": "GET", "url": url},
        "response": {"status": status, "redirectURL": redirect or "", "content": {"mimeType": mime}},
        "_resourceType": rtype,
    }
    if ip:
        e["serverIPAddress"] = ip
    e.update(extra)
    return e


def _har() -> dict:
    return {
        "log": {
            "version": "1.2",
            "pages": [{"title": "Tytuł testowy"}],
            "entries": [
                _entry(
                    "http://track.example.test/r?c=1",
                    302,
                    rtype="document",
                    ip="192.0.2.10",
                    redirect="https://landing.example.test/start",
                ),
                _entry(
                    "https://landing.example.test/start",
                    302,
                    rtype="document",
                    ip="192.0.2.11",
                    redirect="//www.offer.example.test/lp",
                ),  # adres względny (bez schematu)
                _entry(
                    "https://www.offer.example.test/lp",
                    200,
                    rtype="document",
                    ip="198.51.100.7",
                    _securityDetails={"issuer": "Test CA", "sanList": ["offer.example.test", "", 7]},
                ),
                _entry(
                    "https://cdn.example.test/lib.js",
                    200,
                    rtype="script",
                    ip="198.51.100.8",
                    mime="application/javascript",
                ),
                _entry("https://www.googletagmanager.com/gtag/js?id=G-TEST12345", 200, rtype="script"),
                _entry("https://www.googletagmanager.com/gtm.js?id=GTM-ABC123", 200),
                _entry("https://www.facebook.com/tr?id=123456789012345&ev=PageView", 200, rtype="image"),
                _entry("https://tracker.example.test/x.js", 0, rtype="script"),  # błąd sieci
            ],
        }
    }


def test_redirect_chain_follows_document_and_resolves_relative_location():
    seed = har_to_seed(_har())
    assert seed["redirect_chain"] == CHAIN
    assert seed["final_url"] == "https://www.offer.example.test/lp"
    assert seed["http_status"] == 200


def test_hosts_domains_ips_and_scripts():
    seed = har_to_seed(_har())
    assert {"track.example.test", "landing.example.test", "cdn.example.test"} <= set(seed["hosts"])
    assert "example.test" in seed["domains"] and "googletagmanager.com" in seed["domains"]
    assert "example.test" in seed["script_domains"]  # domeny eTLD+1, jak w InfraFeatures.domains
    assert "googletagmanager.com" in seed["script_domains"]
    assert set(seed["ips"]) == {"192.0.2.10", "192.0.2.11", "198.51.100.7", "198.51.100.8"}


def test_tracker_ids_from_request_urls_and_title_and_tls():
    seed = har_to_seed(_har())
    assert seed["gtag"] == ["G-TEST12345"]
    assert seed["gtm"] == ["GTM-ABC123"]
    assert seed["fb_pixel"] == ["123456789012345"]
    assert seed["title"] == "Tytuł testowy"
    assert seed["tls_issuer"] == "Test CA"
    assert seed["tls_sans"] == ["offer.example.test"]  # puste i nietekstowe wpisy odrzucone


def test_network_errors_are_reported_not_raised():
    seed = har_to_seed(_har())
    assert any("bez odpowiedzi" in msg for msg in seed["errors"])


def test_empty_or_unrelated_json_gives_clear_errors(tmp_path):
    assert har_to_seed({"log": {"entries": []}})["errors"] == ["HAR: brak żądań w pliku"]
    with pytest.raises(ValueError, match="nie jest plik HAR"):
        har_to_seed({"nie": "har"})
    bad = tmp_path / "zly.har"
    bad.write_text("{nie-json", encoding="utf-8")
    with pytest.raises(ValueError, match="nie jest poprawnym JSON"):
        read_har(bad)


def test_read_har_file_equals_direct_conversion(tmp_path):
    f = tmp_path / "strona.har"
    f.write_text(json.dumps(_har(), ensure_ascii=False), encoding="utf-8")
    assert read_har(f) == har_to_seed(_har())


def test_seed_feeds_infra_features_offline(tmp_path):
    seed = har_to_seed(_har())
    ctx = ExtractContext(settings=Settings(network=NetworkSettings(enabled=False)), cache_root=tmp_path)
    inf, _ = infra_features(Material(id="h1", source=SourceKind.RECORD, meta={"infra": seed}), ctx)
    assert inf.redirect_chain == CHAIN
    assert inf.final_url == "https://www.offer.example.test/lp" and inf.http_status == 200
    assert {"gtag:G-TEST12345", "gtm:GTM-ABC123", "fbpixel:123456789012345"} <= set(inf.tracker_ids)
    assert "198.51.100.7" in inf.ips
    assert inf.tls_issuer == "Test CA"
    assert inf.title == "Tytuł testowy"


def test_har_in_capture_folder_becomes_one_material_with_infra(tmp_path):
    from PIL import Image

    from slad.ingest import load_dataset

    cap = tmp_path / "p1"  # jak w zbiorze NASK: jeden katalog = jedno przechwycenie kampanii
    cap.mkdir()
    (cap / "strona.html").write_text(
        "<html><head><title>Oferta</title></head><body><p>Zarobisz 5000 zł dziennie</p></body></html>",
        encoding="utf-8",
    )
    Image.new("RGB", (40, 30), "white").save(cap / "zrzut.png")
    (cap / "strona.har").write_text(json.dumps(_har(), ensure_ascii=False), encoding="utf-8")
    mats = load_dataset(tmp_path)
    assert len(mats) == 1
    m = mats[0]
    assert m.html_path and len(m.image_paths) == 1
    assert m.meta["infra"]["redirect_chain"] == CHAIN
    assert m.meta["infra"]["final_url"] == CHAIN[-1]
    assert m.meta.get("source_har") == "strona.har"


def test_broken_har_in_folder_is_a_warning_not_an_error(tmp_path):
    from slad.ingest import load_dataset

    cap = tmp_path / "p2"
    cap.mkdir()
    (cap / "strona.html").write_text("<html><body><p>Tekst reklamy</p></body></html>", encoding="utf-8")
    (cap / "zly.har").write_text("{nie-json", encoding="utf-8")
    mats = load_dataset(tmp_path)
    assert len(mats) == 1 and mats[0].html_path
    assert any("zly.har" in w for w in mats[0].meta.get("ingest_warnings", []))
