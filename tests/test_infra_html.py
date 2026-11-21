"""Analiza HTML: trackery, formularze, linki, favicon, SimHash szablonu, przekierowania klienta, strony „ogólne”."""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.features.infra.html import (  # noqa: E402
    click_tracker_ids,
    client_redirect,
    decode_html_bytes,
    describe_tracker,
    dom_simhash,
    empty_parse,
    extract_tracker_ids,
    generic_page_reason,
    normalize_tracker_id,
    page_url_hint,
    parse_html,
    parse_refresh,
    simhash_distance,
    url_tracker_ids,
)

FIX = Path(__file__).parent / "fixtures" / "infra"
CONTRACT_KEYS = {
    "title",
    "meta_description",
    "text",
    "tracker_ids",
    "script_domains",
    "form_actions",
    "form_fields",
    "outlinks",
    "favicon_url",
    "dom_simhash",
    "html_sha256",
}


def load(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def test_landing_page_features():
    r = parse_html(load("landing_a1.html"), base_url="https://www.wiadomosci24-info.example/art/1")
    assert CONTRACT_KEYS <= set(r)
    assert r["title"] == "Sensacyjne wyznanie: Jan Kowalski zarabia 5000 zł dziennie"
    assert "pasywnego dochodu" in r["meta_description"]
    assert {
        "gtag:G-ABCD123456",
        "aw:AW-123456789",
        "gtm:GTM-AB12CD3",
        "fbpixel:123456789012345",
        "ym:98765432",
        "gsv:gsv-TOKEN-a1b2c3",
        "form:offer_id=OF-7781",
        "form:flow_hash=fl_92ab",
    } <= set(r["tracker_ids"])
    assert len(r["tracker_ids"]) == len(set(r["tracker_ids"]))
    # makro {subid} i utm_* to nie identyfikatory kampanii
    assert not any("subid" in t or "utm_" in t for t in r["tracker_ids"])
    assert r["form_actions"] == [
        "https://crm.leadhub-api.com/api/lead.php",
        "https://www.wiadomosci24-info.example/api/register",  # $.ajax({type: 'POST'}) w skrypcie
    ]
    assert r["form_fields"] == sorted(set(r["form_fields"]))
    assert {"first_name", "last_name", "email", "phone", "offer_id", "flow_hash"} <= set(r["form_fields"])
    assert r["favicon_url"] == "https://www.wiadomosci24-info.example/static/img/fav.png"  # rel=icon przed apple-touch
    assert r["script_domains"] == ["googletagmanager.com", "jsdelivr.net", "trk-analytics.example"]
    assert "onet.pl" in r["outlinks"]
    assert "wiadomosci24-info.example" not in r["outlinks"]  # własna domena nie jest linkiem wychodzącym
    assert {"tel:+48600111222", "mailto:kontakt@wiadomosci24-info.example", "https://wa.me/48600111222"} <= set(
        r["contact_links"]
    )
    text = r["text"]
    assert "Jan Kowalski ujawnia" in text and "Spłaciłem kredyt" in text
    assert "gtag(" not in text and "fbq(" not in text and "Sensacyjne wyznanie" not in text  # bez skryptów i <head>
    assert re.fullmatch(r"[0-9a-f]{64}", r["html_sha256"])
    assert re.fullmatch(r"[0-9a-f]{16}", r["dom_simhash"])
    assert r["generic"] is None and r["client_redirect"] is None
    assert r["lang"] == "pl"


def test_dom_simhash_same_template_vs_different_template():
    a1 = parse_html(load("landing_a1.html"))["dom_simhash"]
    a2 = parse_html(load("landing_a2.html"))["dom_simhash"]  # ten sam szablon, inne teksty, +1 opinia
    b = parse_html(load("landing_b.html"))["dom_simhash"]  # inny szablon
    assert simhash_distance(a1, a2) <= 4
    assert simhash_distance(a1, b) >= 20
    assert simhash_distance(a2, b) >= 20


def test_dom_simhash_ignores_texts_and_attributes():
    html = load("landing_a1.html")
    other_text = re.sub(r">([^<>]{3,})<", lambda m: ">" + "x" * len(m.group(1)) + "<", html)
    other_attrs = re.sub(r'class="[^"]*"', 'class="inna-klasa"', html)
    h = dom_simhash(html)
    assert h is not None
    assert dom_simhash(other_text) == h
    assert dom_simhash(other_attrs) == h


def test_simhash_distance_and_tiny_pages():
    assert simhash_distance("ff", "ff") == 0
    assert simhash_distance("0", "f") == 4
    assert simhash_distance("zz", "00") == 64
    assert simhash_distance(None, "00") == 64
    # strona-zaślepka (kilka elementów) nie dostaje odcisku szablonu — wszystkie zaślepki wyglądałyby tak samo
    assert parse_html("<html><body><p>x</p><script>location.href='/a'</script></body></html>")["dom_simhash"] is None


def test_malformed_html_partial_result():
    r = parse_html((FIX / "malformed.html").read_bytes())
    assert r["title"].startswith("Zepsuta")
    assert "fbpixel:987654321098765" in r["tracker_ids"]
    assert {"phone", "e-mail", "country"} <= set(r["form_fields"])
    assert r["form_actions"] == []  # względny adres bez base_url — nie zgadujemy domeny
    assert r["outlinks"] == ["partner-site.example"]
    assert "tel:+48 500 600 700" in r["contact_links"]
    assert "Tekst bez zamknięcia" in r["text"]
    assert "to nie jest widoczne" not in r["text"]  # niezamknięty komentarz


@pytest.mark.parametrize(
    "junk",
    [
        None,
        "",
        "   ",
        b"\xff\xfe\x00g\x00a\x00r\x00b\x00a\x00g\x00e",
        "<<<>>>",
        "<html",
        "x" * 1000,
        b"<html><body>\x00\x01\x02</body></html>",
        "<div>" * 3000 + "głęboko" + "</div>" * 10,
        "<html><head><title></title><meta http-equiv=refresh content=></head></html>",
        "<a href='http://[zly-ipv6/'>x</a><form action='http://[::1'><input name=a></form>",
        "<script>" + "location.href='" * 50 + "</script>",
    ],
)
def test_parse_html_never_raises(junk):
    r = parse_html(junk, base_url="https://site.example/")
    assert set(empty_parse()) <= set(r)
    assert isinstance(r["text"], str) and len(r["text"]) <= 20_000


def test_text_limit_and_hash_of_bytes():
    html = "<html><body>" + "<p>" + "słowo " * 10_000 + "</p></body></html>"
    r = parse_html(html.encode("utf-8"))
    assert len(r["text"]) == 20_000
    import hashlib

    assert r["html_sha256"] == hashlib.sha256(html.encode("utf-8")).hexdigest()


def test_parked_and_generic_pages():
    assert parse_html(load("parked.html"))["generic"] == "parking domeny"
    assert parse_html("<html><title>Account Suspended</title><body>x</body></html>")["generic"] == (
        "konto hostingu zawieszone"
    )
    assert parse_html("<html><title>Just a moment...</title><body></body></html>")["generic"] == "ochrona przed botami"
    assert parse_html(load("landing_a1.html"))["generic"] is None
    long_page = "<html><body><p>" + "Prawdziwa treść oferty. " * 200 + " page not found </p></body></html>"
    assert parse_html(long_page)["generic"] is None  # długie strony nie są „ogólne”
    assert generic_page_reason({"generic": None}, 404) == "błąd HTTP 404"
    assert generic_page_reason({"generic": "parking domeny"}, 200) == "parking domeny"
    assert generic_page_reason(None, 200) is None


def test_client_redirect_meta_skips_noscript():
    expected = ("https://landing.example/oferta?src=fb", "meta")  # meta refresh z <noscript> jest pomijany
    assert client_redirect(load("redirect_meta.html"), "https://start.example/") == expected
    assert tuple(parse_html(load("redirect_meta.html"), "https://start.example/")["client_redirect"]) == expected


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ('<script>window.location.href = "https://a.example/x";</script>', ("https://a.example/x", "js")),
        ("<script>location.replace('/lp?id=1')</script>", ("https://start.example/lp?id=1", "js")),
        ('<script>top.location = "https:\\/\\/b.example\\/y"</script>', ("https://b.example/y", "js")),
        ("<script>document.location.assign('//c.example/z')</script>", ("https://c.example/z", "js")),
        (
            "<script>window.location.replace(atob('"
            + base64.b64encode(b"https://d.example/q").decode()
            + "'))</script>",
            ("https://d.example/q", "js"),
        ),
        ('<meta http-equiv="refresh" content="0; https://e.example/">', ("https://e.example/", "meta")),
        ('<meta http-equiv="refresh" content="60;url=https://e.example/">', None),  # automatyczne odświeżenie
        ("<button onclick=\"location.href='https://f.example/'\">x</button>", None),  # wymaga kliknięcia
        ('<script src="x.js">location.href="https://g.example/"</script>', None),
        ('<script>location.href = "#top"</script>', None),
        ('<script>location.href = "https://start.example/"</script>', None),  # ta sama strona
        ('<script>location.href = "javascript:void(0)"</script>', None),
        ('<script>if (location.href == "https://h.example/") {}</script>', None),
        ("<p>zwykła strona</p>", None),
        ("", None),
    ],
)
def test_client_redirect_variants(html, expected):
    assert client_redirect(html, "https://start.example/") == expected


def test_js_redirect_only_on_small_pages():
    big = "<html><body>" + "<p>treść</p>" * 5000 + '<script>location.href="https://x.example/"</script></body></html>'
    assert client_redirect(big, "https://start.example/") is None
    assert client_redirect(big, "https://start.example/", allow_js=True) is None


def test_parse_refresh():
    assert parse_refresh("5; url=https://a.example/") == (5.0, "https://a.example/")
    assert parse_refresh("0;URL='/x'") == (0.0, "/x")
    assert parse_refresh("3") == (3.0, None)
    assert parse_refresh("") == (0.0, None)


def test_tracker_ids_only_from_code_not_visible_text():
    html = "<html><body><p>Zegarek G-SHOCK1234 i UA-1234-1 w tekście</p></body></html>"
    assert parse_html(html)["tracker_ids"] == []
    code = parse_html('<script>gtag("config", "G-SHOCK12345")</script>')["tracker_ids"]
    assert code == ["gtag:G-SHOCK12345"]


def test_extract_tracker_ids_variants():
    code = """
      <img src="https://www.facebook.com/tr?id=111122223333444&amp;ev=PageView">
      <script src="https://connect.facebook.net/signals/config/555566667777888?v=2"></script>
      var cfg = {pixel_id: "999988887777666"};
      ttq.load('CABCDEFGHIJ1234567890');
      hj(1,{hjid:3456789,hjsv:6});
      ym(87654321, "init", {});
      <script src="https://mc.yandex.ru/watch/11223344"></script>
      gtag('config', 'GT-ABCDEF12');
      gtag('config', 'AW-987654321/AbC-dEf');
      (window, document, "clarity", "script", "abcdef1234");
      data-ad-client="ca-pub-1234567890123456"
      snaptr('init', '0e4c9b3a-1111-2222-3333-444455556666');
    """
    ids = extract_tracker_ids(code)
    assert {
        "fbpixel:111122223333444",
        "fbpixel:555566667777888",
        "fbpixel:999988887777666",
        "tiktok:CABCDEFGHIJ1234567890",
        "hotjar:3456789",
        "ym:87654321",
        "ym:11223344",
        "gtag:GT-ABCDEF12",
        "aw:AW-987654321",
        "clarity:abcdef1234",
        "adsense:ca-pub-1234567890123456",
        "snap:0e4c9b3a-1111-2222-3333-444455556666",
    } <= set(ids)


def test_click_trackers_and_url_trackers():
    assert click_tracker_ids('<script src="/js/KTracking.js"></script>') == ["tracker:keitaro"]
    assert click_tracker_ids("https://trk.example/?_lp=1&_token=uuid123") == ["tracker:keitaro"]
    assert click_tracker_ids("https://t.example/click.php?lp=1&uclick=x") == ["tracker:binom"]
    assert click_tracker_ids("https://abc.trackvoluum.com/click") == ["tracker:voluum"]
    assert click_tracker_ids("https://x.example/?rtkclickid=1") == ["tracker:redtrack"]
    assert click_tracker_ids("zwykły tekst", None, "") == []
    ids = url_tracker_ids(
        ["https://lp.example/a?pixel=123456789012345&sub1=x", "https://lp.example/b?ttpixel=CABCDEFGHIJ1234567890"]
    )
    assert ids == ["fbpixel:123456789012345", "tiktok:CABCDEFGHIJ1234567890"]
    assert url_tracker_ids(["https://lp.example/a?pixel=12"]) == []  # za krótki — to nie identyfikator piksela


def test_normalize_and_describe_tracker():
    assert normalize_tracker_id("UA-1234-1") == "ga:UA-1234-1"
    assert normalize_tracker_id("g-abc123def4") == "gtag:G-ABC123DEF4"
    assert normalize_tracker_id(" GTM-ABCD ") == "gtm:GTM-ABCD"
    assert normalize_tracker_id("FBPIXEL:123") == "fbpixel:123"
    assert normalize_tracker_id("123456789012345", "fbpixel") == "fbpixel:123456789012345"
    assert normalize_tracker_id("ca-pub-1234567890123456") == "adsense:ca-pub-1234567890123456"
    assert normalize_tracker_id("") is None and normalize_tracker_id(None) is None
    assert describe_tracker("fbpixel:123") == "Facebook Pixel 123"
    assert describe_tracker("tracker:keitaro") == "platforma trackera kliknięć Keitaro"
    assert describe_tracker("nieznany") == "nieznany"


def test_decode_html_bytes():
    text = "Zażółć gęślą jaźń"
    cp = f'<html><head><meta charset="windows-1250"></head><body>{text}</body></html>'.encode("cp1250")
    assert text in decode_html_bytes(cp)
    latin2 = f"<html><body>{text}</body></html>".encode("iso-8859-2")
    assert text in decode_html_bytes(latin2, "text/html; charset=ISO-8859-2")
    assert text in decode_html_bytes("﻿".encode() + text.encode())  # BOM UTF-8
    assert decode_html_bytes(b"\xff\xfe", "text/html; charset=nie-ma-takiego") is not None
    assert decode_html_bytes(b"") == ""


def test_page_url_hint_and_data_uri_favicon():
    saved = "<!-- saved from url=(0029)https://scam.example/oferta -->\n<html><body><a href='/x'>x</a></body></html>"
    assert page_url_hint(saved) == "https://scam.example/oferta"
    r = parse_html(saved)
    assert r["page_url_hint"] == "https://scam.example/oferta"
    assert r["favicon_url"] == "https://scam.example/favicon.ico"
    b = parse_html(load("landing_b.html"))  # bez base_url
    assert b["favicon_url"].startswith("data:image/png;base64,")
    assert b["form_actions"] == ["https://api.quantum-leads.example/v2/leads"]  # fetch(..., {method: 'POST'})
    assert {"tiktok:C9ABCDEF1234567890XY", "clarity:k3x9q2w7ab"} <= set(b["tracker_ids"])
    assert b["title"] == "QuantumAI Pro — Official Trading Platform"
    assert b["meta_description"] == "Join 120,000 members earning with artificial intelligence."
