"""Blokada SSRF (bez sieci i bez prawdziwego DNS): zapis skrócony IPv4 (127.1, 0x7f.1, 2130706433), nazwy domen
wskazujące adresy prywatne (atrapa resolvera), sprawdzenie na każdym przeskoku i połączenie tylko ze sprawdzonym IP."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import httpcore
import httpx
import pytest

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.features.infra import fetch as fetch_mod  # noqa: E402
from slad.features.infra.domains import clean_host, host_of, is_reserved_host  # noqa: E402
from slad.features.infra.fetch import (  # noqa: E402
    BLOCKED_IP,
    BLOCKED_NAME,
    blocked_reason,
    error_message,
    fetch_bytes,
    fetch_chain,
    make_client,
)
from slad.settings import NetworkSettings  # noqa: E402

PUBLIC = "93.184.216.34"
DNS = {
    "127.0.0.1.nip.io": ["127.0.0.1"],
    "169.254.169.254.nip.io": ["169.254.169.254"],
    "localtest.me": ["127.0.0.1", "::1"],
    "mixed.example": [PUBLIC, "10.0.0.5"],  # choć jeden adres niepubliczny wystarczy
    "mapped.example": ["::ffff:127.0.0.1"],
    "scam.example": [PUBLIC],
    "evil.example": [PUBLIC],
    "rebind.example": ["127.0.0.1"],
}


class FakeResolver:
    def __init__(self, table: dict[str, list[str]]):
        self.table = table
        self.calls: list[str] = []

    def __call__(self, host: str) -> list[str]:
        self.calls.append(host)
        return list(self.table.get(host, []))


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("test próbował użyć prawdziwej sieci lub DNS")

    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(fetch_mod, "_RESOLVE_CACHE", {})


@pytest.fixture
def dns(monkeypatch) -> FakeResolver:
    fake = FakeResolver(DNS)
    monkeypatch.setattr(fetch_mod, "resolve_addresses", fake)
    return fake


@pytest.fixture
def connects(monkeypatch) -> list[tuple[str, int]]:
    """Zamiast łączenia: zapis (adres, port), z którym transport chciał się połączyć."""
    seen: list[tuple[str, int]] = []

    def fake_connect(self, host, port, timeout=None, local_address=None, socket_options=None):
        seen.append((host, port))
        raise httpcore.ConnectError("[Errno 61] Connection refused")

    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", fake_connect)
    return seen


@pytest.mark.parametrize(
    ("url", "ip"),
    [
        ("http://127.1/", "127.0.0.1"),
        ("http://127.0.1/", "127.0.0.1"),
        ("http://0x7f.1/", "127.0.0.1"),
        ("http://10.1/", "10.0.0.1"),
        ("http://2130706433/", "127.0.0.1"),  # 127.0.0.1 jako jedna liczba dziesiętna
        ("http://0x7f000001/", "127.0.0.1"),
        ("http://0177.0.0.1/", "127.0.0.1"),  # ósemkowo
        ("http://0xa9.0xfe.0xa9.0xfe/latest/meta-data/", "169.254.169.254"),
    ],
)
def test_ipv4_shorthand_is_blocked(url, ip):
    # adres IP — bez zapytania DNS (autouse odrzuca prawdziwy resolver)
    assert host_of(url) == ip
    assert is_reserved_host(url)
    assert blocked_reason(url) == BLOCKED_IP


def test_clean_host_shorthand_rules():
    assert clean_host("0X7F.1.") == "127.0.0.1"
    assert clean_host("8.8.8.8") == "8.8.8.8"
    assert blocked_reason("http://8.8.8.8/") is None
    # nie jest adresem wg inet_aton — zostaje nazwą (taką nazwę sprawdza resolver)
    for name in ("1.2.3.4.5", "256.1.1.1", "4294967296", "127.08", "1e100.net", "123.example"):
        assert clean_host(name) == name, name


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1.nip.io/admin",
        "http://169.254.169.254.nip.io/latest/meta-data/",
        "http://localtest.me/",
        "https://mixed.example/",
        "http://mapped.example/",
    ],
)
def test_names_resolving_to_private_addresses_are_blocked(dns, url):
    assert blocked_reason(url) == BLOCKED_NAME


def test_name_resolving_to_public_address_is_not_blocked(dns):
    assert blocked_reason("https://scam.example/lp") is None
    assert blocked_reason("https://scam.example/inna") is None
    assert dns.calls.count("scam.example") == 1  # wynik zapamiętany na czas przebiegu
    # nazwa nierozwiązywalna — bez blokady (pobieranie zakończy się błędem DNS)
    assert blocked_reason("https://unresolvable.example/") is None


def test_resolve_addresses_returns_a_and_aaaa(monkeypatch):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        assert host == "dual.example"
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC, 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:2800:220:1::", 0, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC, 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert fetch_mod.resolve_addresses("dual.example") == [PUBLIC, "2606:2800:220:1::"]

    def unknown(*args, **kwargs):
        raise socket.gaierror(8, "nodename nor servname provided, or not known")

    monkeypatch.setattr(socket, "getaddrinfo", unknown)
    assert fetch_mod.resolve_addresses("dead.example") == []


def test_redirect_hop_to_blocked_address_is_blocked(dns):
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        targets = {
            "/a": "http://rebind.example/admin",
            "/b": "http://127.1/",
            "/favicon.ico": "http://169.254.169.254.nip.io/x.ico",
        }
        return httpx.Response(302, headers={"location": targets[request.url.path]})

    t = httpx.MockTransport(handler)
    cfg = NetworkSettings(timeout_s=2)
    r = fetch_chain("https://evil.example/a", cfg, transport=t)
    assert r.error == BLOCKED_NAME
    assert r.chain == ["https://evil.example/a", "http://rebind.example/admin"]
    assert r.final_url == "http://rebind.example/admin" and r.status is None
    r = fetch_chain("https://evil.example/b", cfg, transport=t)
    assert r.error == BLOCKED_IP
    assert fetch_bytes("https://evil.example/favicon.ico", cfg, transport=t)[2] == BLOCKED_NAME
    assert requests == ["https://evil.example/a", "https://evil.example/b", "https://evil.example/favicon.ico"]


def test_connection_goes_only_to_checked_address(monkeypatch, connects):
    answers = iter([[PUBLIC], ["127.0.0.1"]])  # DNS rebinding: druga odpowiedź wskazuje sieć wewnętrzną
    asked: list[str] = []

    def rebinding(host: str) -> list[str]:
        asked.append(host)
        return next(answers)

    monkeypatch.setattr(fetch_mod, "resolve_addresses", rebinding)
    assert blocked_reason("https://rebinding.example/x") is None
    with make_client(NetworkSettings(timeout_s=2)) as client:
        transport = client._transport
        assert isinstance(transport, fetch_mod._PinnedTransport)
        with pytest.raises(httpx.ConnectError) as ei:
            transport.handle_request(httpx.Request("GET", "https://rebinding.example/x"))
    assert connects == [(PUBLIC, 443)]  # sprawdzony adres, nie nowa odpowiedź DNS
    assert asked == ["rebinding.example"]  # sprawdzenie i połączenie widzą ten sam wynik
    assert error_message(ei.value) == "połączenie odrzucone"


def test_connect_time_check_blocks_private_addresses(monkeypatch, connects):
    monkeypatch.setattr(fetch_mod, "resolve_addresses", lambda host: ["10.0.0.7"])
    transport = fetch_mod._PinnedTransport()
    try:
        with pytest.raises(httpx.ConnectError) as ei:
            transport.handle_request(httpx.Request("GET", "http://late-rebind.example/"))
        assert error_message(ei.value) == BLOCKED_NAME
        with pytest.raises(httpx.ConnectError) as ei:
            transport.handle_request(httpx.Request("GET", "http://127.1/"))
        assert error_message(ei.value) == BLOCKED_IP
    finally:
        transport.close()
    assert connects == []
