"""DNS, ASN (Team Cymru) i RDAP — z podstawionym resolwerem i transportem HTTP (bez sieci)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import dns.exception
import dns.resolver
import httpx

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.features.infra import netinfo  # noqa: E402
from slad.features.infra.netinfo import (  # noqa: E402
    asn_for_ip,
    asn_lookup,
    dns_info,
    parse_cymru_asname,
    parse_cymru_origin,
    parse_rdap_domain,
    rdap_domain,
    rdap_ip,
    to_iso,
)


class FakeResolver:
    """Udaje dnspython: (nazwa, typ) → lista rekordów albo wyjątek; zapisuje zapytania."""

    def __init__(self, table: dict[tuple[str, str], object]):
        self.table = table
        self.queries: list[tuple[str, str]] = []

    def resolve(self, name, rdtype, lifetime=None, **kw):
        self.queries.append((str(name), str(rdtype)))
        value = self.table.get((str(name), str(rdtype)))
        if value is None:
            raise dns.resolver.NoAnswer()
        if isinstance(value, BaseException) or (isinstance(value, type) and issubclass(value, BaseException)):
            raise value
        return value


def a(addr):
    return SimpleNamespace(address=addr)


def target(name):
    return SimpleNamespace(target=name)


def mx(pref, name):
    return SimpleNamespace(preference=pref, exchange=name)


def txt(text):
    return SimpleNamespace(strings=(text.encode(),))


def test_dns_info_records_and_zone_queries():
    res = FakeResolver(
        {
            ("www.scam-news.pl", "A"): [a("104.21.5.6"), a("172.67.1.2"), a("104.21.5.6")],
            ("www.scam-news.pl", "AAAA"): [a("2606:4700::1")],
            ("www.scam-news.pl", "CNAME"): [target("Scam-News.pl.")],
            ("scam-news.pl", "NS"): [target("Kate.NS.Cloudflare.com."), target("bob.ns.cloudflare.com.")],
            ("scam-news.pl", "MX"): [mx(20, "mx2.mail.example."), mx(10, "MX1.mail.example.")],
        }
    )
    out = dns_info("https://www.scam-news.pl/art", timeout=1, resolver=res)
    assert out["a"] == ["104.21.5.6", "172.67.1.2"]
    assert out["aaaa"] == ["2606:4700::1"]
    assert out["cname"] == ["scam-news.pl"]
    assert out["ns"] == ["bob.ns.cloudflare.com", "kate.ns.cloudflare.com"]  # małe litery, bez kropki, posortowane
    assert out["mx"] == ["mx1.mail.example", "mx2.mail.example"]  # wg priorytetu
    assert out["errors"] == []
    assert ("scam-news.pl", "NS") in res.queries  # NS/MX dla domeny zarejestrowanej, nie hosta


def test_dns_info_nxdomain_and_timeout():
    res = FakeResolver(
        {
            ("dead.example.com", "A"): dns.resolver.NXDOMAIN(),
            ("example.com", "NS"): [target("a.iana-servers.net.")],
        }
    )
    out = dns_info("dead.example.com", timeout=1, resolver=res)
    assert out["a"] == [] and out["errors"] == ["dns: domena nie istnieje"]
    assert ("dead.example.com", "AAAA") not in res.queries  # po NXDOMAIN nie pytamy dalej o ten host
    assert out["ns"] == ["a.iana-servers.net"]

    slow = FakeResolver({("slow.pl", "A"): dns.exception.Timeout(), ("slow.pl", "NS"): dns.resolver.NoNameservers()})
    out = dns_info("slow.pl", timeout=1, resolver=slow)
    assert "dns: przekroczono czas" in out["errors"]
    assert "dns: serwery nazw nie odpowiadają" in out["errors"]


def test_dns_info_ip_and_invalid_host():
    res = FakeResolver({})
    assert dns_info("8.8.8.8", resolver=res)["a"] == ["8.8.8.8"]
    assert dns_info("http://[2001:db8::1]/", resolver=res)["aaaa"] == ["2001:db8::1"]
    assert res.queries == []
    assert dns_info("", resolver=res)["errors"] == ["dns: nieprawidłowa nazwa hosta"]
    assert dns_info("exa mple", resolver=res)["errors"] == ["dns: nieprawidłowa nazwa hosta"]


def test_dns_info_unexpected_exception_never_raises():
    res = FakeResolver({("x.pl", "A"): RuntimeError("awaria")})
    out = dns_info("x.pl", timeout=1, resolver=res)
    assert out["errors"] and out["errors"][0].startswith("dns: błąd")


def test_cymru_parsers():
    assert parse_cymru_origin(["13335 | 104.16.0.0/13 | US | arin | 2014-03-28"]) == "13335"
    # kilka prefiksów → najdłuższy (najbardziej szczegółowy) wygrywa
    assert parse_cymru_origin(["15169 | 8.0.0.0/9 | US | arin |", "3356 | 8.8.8.0/24 | US | arin |"]) == "3356"
    assert parse_cymru_origin(["13335 209242 | 1.1.1.0/24 | AU | apnic |"]) == "13335"
    assert parse_cymru_origin(["śmieci"]) is None and parse_cymru_origin([]) is None
    assert (
        parse_cymru_asname(["13335 | US | arin | 2010-07-14 | CLOUDFLARENET - Cloudflare, Inc., US"]) == "CLOUDFLARENET"
    )
    assert parse_cymru_asname(["16276 | FR | ripencc | 2001-02-15 | OVH, FR"]) == "OVH"
    assert parse_cymru_asname(["24940 | DE | ripencc | | HETZNER-AS, DE"]) == "HETZNER-AS"
    assert parse_cymru_asname(["krótkie | pole"]) is None


def test_asn_lookup_with_fake_resolver():
    netinfo._AS_NAMES.clear()
    res = FakeResolver(
        {
            ("6.5.21.104.origin.asn.cymru.com", "TXT"): [txt("13335 | 104.21.0.0/16 | US | arin | 2014-03-28")],
            ("AS13335.asn.cymru.com", "TXT"): [
                txt("13335 | US | arin | 2010-07-14 | CLOUDFLARENET - Cloudflare, Inc., US")
            ],
        }
    )
    assert asn_lookup("104.21.5.6", timeout=1, resolver=res) == ("AS13335 CLOUDFLARENET", None)
    assert asn_for_ip("104.21.5.6", timeout=1, resolver=res) == "AS13335 CLOUDFLARENET"
    assert res.queries.count(("AS13335.asn.cymru.com", "TXT")) == 1  # nazwa AS zapamiętana w procesie
    # adres prywatny — brak ASN, ale to nie błąd; śmieci → błąd
    assert asn_lookup("192.168.1.1", resolver=res) == (None, None)
    assert asn_lookup("nie-ip", resolver=res) == (None, "asn: nieprawidłowy adres IP")
    # brak danych i przekroczony czas
    assert asn_lookup("9.9.9.9", timeout=1, resolver=res) == (None, "asn: brak danych")
    slow = FakeResolver({("1.1.1.1.origin.asn.cymru.com", "TXT"): dns.exception.Timeout()})
    assert asn_lookup("1.1.1.1", timeout=1, resolver=slow) == (None, "asn: przekroczono czas")


def test_asn_lookup_ipv6_query_name():
    res = FakeResolver({})
    asn_lookup("2606:4700::1111", timeout=1, resolver=res)
    name = res.queries[0][0]
    assert name.endswith(".origin6.asn.cymru.com")
    assert name.startswith("1.1.1.1.0.0.0.0")  # odwrócone półbajty adresu


RDAP_DOMAIN = {
    "objectClassName": "domain",
    "ldhName": "SCAM-NEWS.PL",
    "status": ["active", "client transfer prohibited"],
    "events": [
        {"eventAction": "registration", "eventDate": "2025-10-01T12:30:00Z"},
        {"eventAction": "expiration", "eventDate": "2026-10-01T12:30:00Z"},
        {"eventAction": "last changed", "eventDate": "2025-10-02T08:00:00+02:00"},
    ],
    "nameservers": [{"ldhName": "NS2.DNSOWL.COM."}, {"ldhName": "ns1.dnsowl.com"}, {"unicodeName": "ns3.dnsowl.com"}],
    "entities": [
        {"roles": ["registrant"], "vcardArray": ["vcard", [["fn", {}, "text", "REDACTED"]]]},
        {
            "roles": ["registrar"],
            "handle": "1479",
            "vcardArray": ["vcard", [["version", {}, "text", "4.0"], ["fn", {}, "text", "NameSilo, LLC"]]],
            "entities": [{"roles": ["abuse"], "vcardArray": ["vcard", [["fn", {}, "text", "Abuse"]]]}],
        },
    ],
}


def test_parse_rdap_domain():
    out = parse_rdap_domain(RDAP_DOMAIN)
    assert out["registrar"] == "NameSilo, LLC"
    assert out["created"] == "2025-10-01T12:30:00+00:00"
    assert out["updated"] == "2025-10-02T06:00:00+00:00"  # strefa → UTC
    assert out["expires"] == "2026-10-01T12:30:00+00:00"
    assert out["nameservers"] == ["ns1.dnsowl.com", "ns2.dnsowl.com", "ns3.dnsowl.com"]
    assert out["status"] == ["active", "client transfer prohibited"]
    # rejestrator zagnieżdżony głębiej i tylko z identyfikatorem
    nested = {"entities": [{"roles": ["registrant"], "entities": [{"roles": ["registrar"], "handle": "292"}]}]}
    assert parse_rdap_domain(nested)["registrar"] == "292"
    assert parse_rdap_domain(None)["registrar"] is None
    assert parse_rdap_domain({"events": "zły typ", "nameservers": [None, 5]})["nameservers"] == []


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_rdap_domain_http():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/domain/scam-news.pl":
            return httpx.Response(200, json=RDAP_DOMAIN)
        if request.url.path == "/domain/missing.pl":
            return httpx.Response(404)
        if request.url.path == "/domain/limit.pl":
            return httpx.Response(429)
        if request.url.path == "/domain/slow.pl":
            raise httpx.ReadTimeout("timeout")
        return httpx.Response(200, content=b"<html>nie json</html>")

    with make_client(handler) as c:
        out = rdap_domain("https://www.scam-news.pl/art?x=1", timeout=1, client=c)
        assert seen[0] == "https://rdap.org/domain/scam-news.pl"  # domena zarejestrowana z adresu
        assert out["registrar"] == "NameSilo, LLC" and out["error"] is None
        assert rdap_domain("missing.pl", client=c)["error"] == "rdap: brak danych (HTTP 404)"
        assert rdap_domain("limit.pl", client=c)["error"] == "rdap: limit zapytań (HTTP 429)"
        assert rdap_domain("slow.pl", client=c)["error"] == "rdap: przekroczono czas"
        assert rdap_domain("html.pl", client=c)["error"] == "rdap: nieprawidłowa odpowiedź"
        assert rdap_domain("", client=c)["error"] == "rdap: nieprawidłowa domena"


def test_rdap_ip():
    payload = {
        "name": "CLOUDFLARENET",
        "handle": "NET-104-16-0-0-1",
        "country": "US",
        "cidr0_cidrs": [{"v4prefix": "104.16.0.0", "length": 12}],
        "arin_originas0_originautnums": [13335],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ip/104.21.5.6"
        return httpx.Response(200, content=json.dumps(payload).encode())

    with make_client(handler) as c:
        out = rdap_ip("104.21.5.6", client=c)
    assert out == {
        "name": "CLOUDFLARENET",
        "handle": "NET-104-16-0-0-1",
        "country": "US",
        "cidr": "104.16.0.0/12",
        "asn": "AS13335",
        "error": None,
    }
    assert rdap_ip("zly-adres")["error"] == "rdap: nieprawidłowy adres IP"


def test_to_iso():
    assert to_iso("2025-10-01T12:30:00Z") == "2025-10-01T12:30:00+00:00"
    assert to_iso("2025-10-01") == "2025-10-01T00:00:00+00:00"
    assert to_iso("wczoraj") is None and to_iso(None) is None and to_iso("") is None
