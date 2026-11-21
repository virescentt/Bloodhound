"""Domeny i adresy: domena zarejestrowana (offline PSL), host, normalizacja, opakowania przekierowań, listy wyjątków."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.features.infra import domains  # noqa: E402
from slad.features.infra.domains import (  # noqa: E402
    host_of,
    is_platform_url,
    is_reserved_host,
    is_stoplisted_domain,
    load_stoplists,
    normalize_url,
    prepare_fetch_url,
    refang,
    registered_domain,
    unique,
    unwrap_redirector,
)


def test_tldextract_is_offline():
    # migawka PSL z pakietu — nigdy nie pobieramy listy z sieci
    assert tuple(domains._EXTRACT.suffix_list_urls) == ()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://WWW.Example.COM/path", "example.com"),
        ("sub.example.co.uk", "example.co.uk"),
        ("forums.news.cnn.co.uk", "cnn.co.uk"),
        ("http://żółw.pl/x", "xn--w-uga1v8h.pl"),  # IDN → punycode
        ("xn--w-uga1v8h.pl", "xn--w-uga1v8h.pl"),
        ("https://foo.netlify.app/", "foo.netlify.app"),  # prywatna część PSL: osobny właściciel
        ("https://bar.github.io/x", "bar.github.io"),
        ("https://scam.weebly.com", "scam.weebly.com"),  # darmowy kreator spoza PSL
        ("netlify.app", "netlify.app"),
        ("a.b.c.unknowntld", "c.unknowntld"),  # nieznana TLD — ostatnie dwie etykiety
        ("hxxps://scam[.]pl/a", "scam.pl"),  # adres „rozbrojony”
        ("example.com.", "example.com"),
        ("http://127.0.0.1:8080/", None),
        ("http://[2001:db8::1]/", None),
        ("10.0.0.1", None),
        ("brak", None),
        ("localhost", None),
        ("", None),
        (None, None),
        ("mailto:a@b.pl", None),
        ("http://exa mple.com", None),
    ],
)
def test_registered_domain(value, expected):
    assert registered_domain(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://User:Pass@Sub.Example.com:8443/a?b#c", "sub.example.com"),
        ("example.com/path", "example.com"),
        ("//cdn.example.org/x.js", "cdn.example.org"),
        ("http://[2001:DB8::1]:80/", "2001:db8::1"),
        ("http://żółw.pl", "xn--w-uga1v8h.pl"),
        ("javascript:alert(1)", None),
        ("  ", None),
    ],
)
def test_host_of(value, expected):
    assert host_of(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "Example.com/Path?utm_source=fb&utm_medium=cpc&a=1&fbclid=X&gclid=Y#frag",
            "http://example.com/Path?a=1",
        ),
        ("https://example.com/?yclid=1&mc_eid=2&_ga=3&keep=4&UTM_Campaign=z", "https://example.com/?keep=4"),
        ("HTTPS://WWW.EXAMPLE.COM:443", "https://www.example.com/"),
        ("http://example.com:80/a", "http://example.com/a"),
        ("http://example.com:8080", "http://example.com:8080/"),
        ("https://user:pw@example.com/a b", "https://example.com/a%20b"),
        ("//cdn.example.org/x.js", "http://cdn.example.org/x.js"),
        ("http://żółw.pl/ą", "http://xn--w-uga1v8h.pl/%C4%85"),
        ("hxxps://scam[.]pl/lp?sub1=abc", "https://scam.pl/lp?sub1=abc"),
        ("https://example.com/lp?pixel=123&flag", "https://example.com/lp?pixel=123&flag"),
        ("mailto:a@b.pl", None),
        ("javascript:alert(1)", None),
        ("brak", None),
        ("N/A", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_url(value, expected):
    assert normalize_url(value) == expected


def test_prepare_fetch_url_keeps_parameters():
    # do pobrania zostawiamy parametry (część cloakerów ich wymaga), usuwamy tylko fragment i dane logowania
    assert prepare_fetch_url("scam.pl/lp?fbclid=1&sub=2#x") == "http://scam.pl/lp?fbclid=1&sub=2"
    assert prepare_fetch_url("https://u:p@Scam.PL:443/") == "https://scam.pl/"
    assert prepare_fetch_url("brak") is None
    assert prepare_fetch_url("tg://resolve?domain=x") is None


def test_refang_and_unique():
    assert refang("hxxp://evil[.]com[/]a") == "http://evil.com/a"
    assert refang("hXXps[:]//evil(.)pl") == "https://evil.pl"
    assert unique(["a", None, "b", "a", "", "c", "b"]) == ["a", "b", "c"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "https://l.facebook.com/l.php?u=https%3A%2F%2Fscam.example%2Flp%3Fa%3D1&h=AT0",
            "https://scam.example/lp?a=1",
        ),
        ("https://lm.facebook.com/l.php?u=http%3A%2F%2Fscam.pl%2F", "http://scam.pl/"),
        ("https://l.instagram.com/?u=https%3A%2F%2Fscam.pl%2Fx&e=1", "https://scam.pl/x"),
        ("https://www.google.com/url?q=https://scam.pl/a&sa=D", "https://scam.pl/a"),
        ("https://www.google.pl/url?url=https%3A%2F%2Fscam.pl%2F", "https://scam.pl/"),
        ("https://www.youtube.com/redirect?q=https%3A%2F%2Fscam.pl%2Fy&v=1", "https://scam.pl/y"),
        ("https://vk.com/away.php?to=https%3A%2F%2Fscam.pl", "https://scam.pl/"),
        ("https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fscam.pl%2Fz&data=1", "https://scam.pl/z"),
        ("https://www.facebook.com/somepage", None),
        ("https://scam.pl/l.php?u=https://other.pl", None),
        ("https://l.facebook.com/l.php?u=javascript:alert(1)", None),
        (None, None),
    ],
)
def test_unwrap_redirector(value, expected):
    assert unwrap_redirector(value) == expected


def test_platform_and_reserved_hosts():
    assert is_platform_url("https://www.facebook.com/ads/library/?id=1")
    assert is_platform_url("https://t.me/kanal")
    assert is_platform_url("https://vm.tiktok.com/abc")
    assert not is_platform_url("https://bit.ly/abc")  # skracacz — trzeba przejść
    assert not is_platform_url("https://sites.google.com/view/oszustwo")  # hosting na Google Sites — pobieramy
    assert not is_platform_url("https://scam.pl")
    assert is_reserved_host("https://scam.example/")
    assert is_reserved_host("http://x.test")
    assert is_reserved_host("localhost")
    assert is_reserved_host("http://192.168.1.10/")
    assert not is_reserved_host("https://scam.pl")
    assert not is_reserved_host("http://8.8.8.8/")


def test_stoplists_from_config():
    lists = load_stoplists()
    assert set(lists) == {"domains", "asns", "nameserver_patterns", "registrars", "tls_issuers", "form_fields"}
    assert "googletagmanager.com" in lists["domains"]
    assert "as13335" in lists["asns"]
    assert any("let's encrypt" in x for x in lists["tls_issuers"])
    assert is_stoplisted_domain("cdn.jsdelivr.net")  # subdomena pozycji listy
    assert is_stoplisted_domain("https://www.facebook.com/x")
    assert not is_stoplisted_domain("notfacebook.com")
    assert not is_stoplisted_domain("scam.pl")


def test_stoplists_missing_or_broken_file(tmp_path):
    assert load_stoplists(str(tmp_path / "brak.yaml"))["domains"] == ()
    broken = tmp_path / "zly.yaml"
    broken.write_text("domains: [niezamknięta", encoding="utf-8")
    assert load_stoplists(str(broken))["domains"] == ()
    custom = {"domains": ("scam-cdn.example",)}
    assert is_stoplisted_domain("a.scam-cdn.example", custom)
