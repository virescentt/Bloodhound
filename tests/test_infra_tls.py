"""Certyfikaty TLS (lokalny serwer TLS na 127.0.0.1 z wygenerowanym certyfikatem) i crt.sh (MockTransport)."""

from __future__ import annotations

import hashlib
import socket
import ssl
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.features.infra.tls import cert_info, crtsh_domains, crtsh_lookup, parse_crtsh, tls_info  # noqa: E402


@pytest.fixture(scope="module")
def cert_pair(tmp_path_factory):
    """Samopodpisany certyfikat „oszusta” z kilkoma nazwami SAN (w tym innej domeny)."""
    key = ec.generate_private_key(ec.SECP256R1())
    issuer = x509.Name(
        [x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Fake CA Org"), x509.NameAttribute(NameOID.COMMON_NAME, "R99")]
    )
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "scam-news.example")])
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=89))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("scam-news.example"),
                    x509.DNSName("*.scam-news.example"),
                    x509.DNSName("Other-Scam.example"),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    d = tmp_path_factory.mktemp("tls")
    cert_file, key_file = d / "cert.pem", d / "key.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    return cert, cert_file, key_file


def test_cert_info_from_der(cert_pair):
    cert, _, _ = cert_pair
    der = cert.public_bytes(serialization.Encoding.DER)
    info = cert_info(der)
    assert info["error"] is None
    assert info["fingerprint"] == hashlib.sha256(der).hexdigest()
    assert info["issuer"] == "Fake CA Org (R99)"
    assert info["subject"] == "scam-news.example"
    assert info["sans"] == ["scam-news.example", "other-scam.example"]  # bez *., małe litery, bez powtórzeń
    assert info["not_before"] < info["not_after"]
    datetime.fromisoformat(info["not_after"])


def test_cert_info_garbage():
    info = cert_info(b"to nie jest certyfikat")
    assert info["fingerprint"] == hashlib.sha256(b"to nie jest certyfikat").hexdigest()
    assert info["error"].startswith("tls: nie można odczytać certyfikatu")
    assert cert_info(b"")["error"] == "tls: brak certyfikatu"


def test_tls_info_against_local_server(cert_pair):
    cert, cert_file, key_file = cert_pair
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(cert_file, key_file)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5)
    port = listener.getsockname()[1]

    def serve() -> None:
        try:
            conn, _ = listener.accept()
            with server_ctx.wrap_socket(conn, server_side=True) as tls_conn:
                tls_conn.settimeout(5)
                try:
                    tls_conn.recv(1)
                except (OSError, ssl.SSLError):
                    pass
        except (OSError, ssl.SSLError):
            pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        info = tls_info("127.0.0.1", port, timeout=5)  # certyfikat samopodpisany — i tak go odczytujemy
    finally:
        thread.join(timeout=5)
        listener.close()
    assert info["error"] is None
    assert info["fingerprint"] == hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
    assert info["issuer"] == "Fake CA Org (R99)"
    assert "other-scam.example" in info["sans"]


def test_tls_info_errors():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nic nie nasłuchuje
    info = tls_info("127.0.0.1", port, timeout=2)
    assert info["error"].startswith("tls: błąd połączenia") and info["fingerprint"] is None
    assert tls_info("", timeout=1)["error"] == "tls: nieprawidłowa nazwa hosta"
    assert tls_info("exa mple", timeout=1)["error"] == "tls: nieprawidłowa nazwa hosta"


CRTSH = [
    {"common_name": "scam-news.example", "name_value": "scam-news.example\n*.scam-news.example"},
    {
        "common_name": "www.scam-news.example",
        "name_value": "www.scam-news.example\nother-scam.example\nOTHER-SCAM.example",
    },
    {"common_name": "cpanel.scam-news.example", "name_value": "mail.scam-news.example\nfoo@bar\n"},
    "zły rekord",
]


def test_parse_crtsh():
    assert parse_crtsh(CRTSH) == [
        "scam-news.example",
        "www.scam-news.example",
        "other-scam.example",
        "cpanel.scam-news.example",
        "mail.scam-news.example",
    ]
    many = [{"name_value": f"h{i}.example.com"} for i in range(500)]
    assert len(parse_crtsh(many)) == 200
    assert parse_crtsh({"error": "x"}) == []


def test_crtsh_http():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("q") == "%.scam-news.example":
            assert request.url.params.get("output") == "json"
            return httpx.Response(200, json=CRTSH)
        if request.url.params.get("q") == "%.slow.example":
            raise httpx.ReadTimeout("timeout")
        return httpx.Response(502)

    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        assert "other-scam.example" in crtsh_domains("https://www.scam-news.example/x", timeout=1, client=c)
        assert crtsh_lookup("down.example", timeout=1, client=c) == ([], "crt.sh: HTTP 502")
        assert crtsh_lookup("slow.example", timeout=1, client=c) == ([], "crt.sh: przekroczono czas")
        assert crtsh_domains("down.example", timeout=1, client=c) == []
        assert crtsh_lookup("", client=c) == ([], "crt.sh: nieprawidłowa domena")
