"""Skrót favicony zgodny z Shodanem, rozpoznawanie obrazów, adresy data:."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import mmh3

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.features.infra.favicon import decode_data_uri, favicon_hash, favicon_hash_from_data_uri, is_image  # noqa: E402

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def test_known_vector():
    # base64.encodebytes(b"hello") == b"aGVsbG8=\n" (z końcowym znakiem nowej linii, jak w Shodanie)
    assert favicon_hash(b"hello") == mmh3.hash(b"aGVsbG8=\n") == 1155597304


def test_shodan_style_line_breaks_and_signed_int():
    data = bytes(range(256)) * 3  # > 57 bajtów → base64 łamany co 76 znaków
    h = favicon_hash(data)
    assert h == mmh3.hash(base64.encodebytes(data))
    assert h != mmh3.hash(base64.b64encode(data))  # bez łamania wierszy wynik jest inny
    assert -(2**31) <= h < 2**31
    assert favicon_hash(PNG_1PX) == favicon_hash(bytearray(PNG_1PX))
    assert any(favicon_hash(bytes([i])) < 0 for i in range(256))  # wynik ze znakiem (int32), jak w Shodanie


def test_is_image():
    assert is_image(PNG_1PX)
    assert is_image(b"\x00\x00\x01\x00\x01\x00\x10\x10")  # ICO
    assert is_image(b"GIF89a....")
    assert is_image(b"\xff\xd8\xff\xe0....")
    assert is_image(b"RIFF\x00\x00\x00\x00WEBPVP8 ")
    assert is_image(b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"></svg>')
    assert is_image(b"<svg></svg>")
    # strona błędu HTML zwrócona pod /favicon.ico to NIE favicona (byłaby wspólna dla całego hostingu)
    assert not is_image(b"<!DOCTYPE html><html><body>404</body></html>", "image/x-icon")
    assert not is_image(b"<html>", "text/html")
    assert not is_image(b"", "image/png")
    assert not is_image(None)


def test_data_uri():
    uri = "data:image/png;base64," + base64.b64encode(PNG_1PX).decode()
    assert decode_data_uri(uri) == PNG_1PX
    assert favicon_hash_from_data_uri(uri) == favicon_hash(PNG_1PX)
    assert decode_data_uri("data:image/svg+xml,%3Csvg%3E%3C%2Fsvg%3E") == b"<svg></svg>"
    assert favicon_hash_from_data_uri("data:text/html,<html>nie obraz</html>") is None
    assert decode_data_uri("https://example.com/favicon.ico") is None
    assert decode_data_uri(None) is None
    assert decode_data_uri("data:zepsute") is None
