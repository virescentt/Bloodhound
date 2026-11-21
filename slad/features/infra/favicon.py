"""Skrót favicony zgodny z Shodanem (``http.favicon.hash``): MurmurHash3 (32 bity, ze znakiem) z base64 treści
z podziałem na wiersze po 76 znaków — dokładnie tak, jak liczy go Shodan, więc wartość można wyszukać w Shodanie.

Ta sama favicona na różnych domenach to częsty ślad jednego zestawu (kitu) stron oszustów.
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
from urllib.parse import unquote_to_bytes

import mmh3

log = logging.getLogger(__name__)

_DATA_URI_RE = re.compile(r"^data:([^,;]*)((?:;[^,;]*)*?)(;base64)?,(.*)$", re.I | re.S)


def favicon_hash(content: bytes) -> int:
    """Skrót favicony w stylu Shodana: ``mmh3.hash(base64.encodebytes(content))`` (int32 ze znakiem)."""
    return mmh3.hash(base64.encodebytes(bytes(content)))


def is_image(content: bytes | None, content_type: str | None = None) -> bool:
    """Czy treść to obraz (ICO, PNG, GIF, JPEG, WebP, BMP, SVG)? Chroni przed liczeniem skrótu ze strony błędu
    HTML zwróconej pod /favicon.ico — taka „favicona” jest wspólna dla całego hostingu, nie dla kampanii."""
    if not content or len(content) < 4:
        return False
    head = content[:512]
    if head.startswith((b"\x00\x00\x01\x00", b"\x00\x00\x02\x00", b"\x89PNG", b"GIF8", b"\xff\xd8\xff", b"BM")):
        return True
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return True
    low = head.lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if low.startswith(b"<svg") or (low.startswith(b"<?xml") and b"<svg" in content[:4096].lower()):
        return True
    if low.startswith((b"<!doctype", b"<html", b"<head", b"<body")):
        return False
    ct = (content_type or "").split(";")[0].strip().lower()
    return ct.startswith("image/") and not ct.startswith("image/svg") and b"<html" not in low[:200]


def decode_data_uri(uri: str | None) -> bytes | None:
    """Treść z adresu ``data:image/png;base64,…`` (favicona osadzona w HTML — dostępna bez sieci)."""
    if not uri or not uri[:5].lower() == "data:":
        return None
    m = _DATA_URI_RE.match(uri.strip())
    if not m:
        return None
    payload = m.group(4)
    try:
        if m.group(3):
            return base64.b64decode(re.sub(r"\s+", "", unquote_to_bytes(payload).decode("ascii", "ignore")) + "==")
        return unquote_to_bytes(payload)
    except (binascii.Error, ValueError) as e:
        log.debug("data URI: %s", e)
        return None


def favicon_hash_from_data_uri(uri: str | None) -> int | None:
    data = decode_data_uri(uri)
    if data and is_image(data, uri.split(";", 1)[0][5:] if uri else None):
        return favicon_hash(data)
    return None
