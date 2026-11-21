"""Pobieranie strony z pełnym łańcuchem przekierowań: HTTP 3xx, nagłówek ``Refresh``, ``<meta http-equiv=refresh>``
i proste przekierowania JavaScript (rozpoznawane wzorcem tekstowym na małych stronach — JS nie jest wykonywany).

Zasady bezpieczeństwa (pobieramy strony oszustów):
- tylko GET, bez JavaScriptu; limit przeskoków (``max_redirects``), wykrywanie pętli,
- limit rozmiaru (``max_bytes``, odczyt strumieniowy) i czasu (``timeout_s`` na operację + łączny limit łańcucha),
- treść dekodowana wyłącznie dla text/html, application/xhtml+xml i text/plain (brak typu → rozpoznanie po
  pierwszych bajtach); pliki binarne nie są czytane, zapisywane ani uruchamiane,
- adresy prywatne i lokalne (127.0.0.1, 10.x, 169.254.x, localhost…) są blokowane (ochrona przed SSRF),
- błąd certyfikatu TLS → jedna ponowna próba bez weryfikacji i znacznik ``tls_invalid``.
Funkcje nigdy nie rzucają wyjątków — błąd trafia do pola ``error`` (krótko, po polsku).
"""

from __future__ import annotations

import ipaddress
import logging
import ssl
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import httpx

from slad.features.infra.domains import host_of, is_ip, normalize_url, prepare_fetch_url
from slad.features.infra.html import client_redirect, decode_html_bytes, parse_refresh

log = logging.getLogger(__name__)

TEXT_TYPES = frozenset({"text/html", "application/xhtml+xml", "text/plain"})
ACCEPT_HTML = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
ACCEPT_IMAGE = "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
ACCEPT_LANGUAGE = "pl-PL,pl;q=0.9,en-US;q=0.6,en;q=0.5"
REFRESH_MAX_DELAY_S = 10.0
FAVICON_MAX_BYTES = 1_000_000
_CHUNK = 65_536


@dataclass
class FetchResult:
    chain: list[str] = field(default_factory=list)  # kolejne adresy: wejściowy … końcowy
    final_url: str | None = None  # ostatni adres, o który pytaliśmy (także gdy nie odpowiedział)
    status: int | None = None  # kod HTTP ostatniej odpowiedzi (None — brak odpowiedzi)
    headers: dict[str, str] = field(default_factory=dict)  # nagłówki ostatniej odpowiedzi (małe litery)
    body: str | None = None  # zdekodowany HTML/tekst strony końcowej (None — brak / plik binarny)
    content_type: str | None = None
    error: str | None = None
    tls_invalid: bool = False  # certyfikat nieprawidłowy — pobrano bez weryfikacji
    # poza kontraktem:
    pages: list[tuple[str, str]] = field(default_factory=list)  # (adres, HTML) każdej strony HTML w łańcuchu
    hops: list[dict[str, Any]] = field(default_factory=list)  # [{"url", "status", "via": start|http|refresh|meta|js}]
    truncated: bool = False  # treść ucięta na max_bytes
    elapsed_s: float = 0.0


@dataclass
class _Hop:
    status: int
    headers: dict[str, str]
    content_type: str | None
    location: str | None
    body: str | None
    truncated: bool = False


# ---------------------------------------------------------------- klient


def request_headers(cfg: Any, accept: str = ACCEPT_HTML) -> dict[str, str]:
    ua = getattr(cfg, "user_agent", None) or "Mozilla/5.0"
    return {"User-Agent": ua, "Accept": accept, "Accept-Language": ACCEPT_LANGUAGE}


def make_client(cfg: Any, *, verify: bool = True, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """Klient HTTP bez automatycznych przekierowań (łańcuch budujemy sami)."""
    return httpx.Client(
        follow_redirects=False,
        timeout=httpx.Timeout(float(getattr(cfg, "timeout_s", 8.0))),
        headers=request_headers(cfg),
        verify=verify,
        transport=transport,
    )


def _is_ssl_error(exc: BaseException) -> bool:
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, ssl.SSLError):
            return True
        msg = str(e).lower()
        if any(k in msg for k in ("ssl", "certificate", "tlsv", "handshake")):
            return True
        e = e.__cause__ or e.__context__
    return False


_DNS_MARKERS = (
    "name or service not known",
    "nodename nor servname",
    "getaddrinfo",
    "no address associated",
    "name resolution",
    "[errno 8]",
    "[errno -2]",
    "[errno -3]",
    "[errno -5]",
    "[errno 11001]",
)


def error_message(exc: BaseException) -> str:
    """Wyjątek sieciowy → krótki komunikat po polsku."""
    if isinstance(exc, httpx.TimeoutException):
        return "przekroczono czas"
    if isinstance(exc, httpx.ConnectError):
        msg = str(exc).lower()
        if any(k in msg for k in _DNS_MARKERS):
            return "nie można rozwiązać nazwy domeny"
        if "refused" in msg:
            return "połączenie odrzucone"
        if _is_ssl_error(exc):
            return "błąd TLS/SSL"
        return "błąd połączenia"
    if isinstance(exc, httpx.RemoteProtocolError):
        return "błąd protokołu HTTP"
    if isinstance(exc, httpx.UnsupportedProtocol):
        return "nieobsługiwany protokół"
    if isinstance(exc, (httpx.InvalidURL, ValueError, UnicodeError)):
        return "nieprawidłowy adres"
    if isinstance(exc, httpx.HTTPError):
        return f"błąd HTTP ({type(exc).__name__})"
    return f"błąd ({type(exc).__name__})"


def blocked_reason(url: str) -> str | None:
    """Adresy, których nie pobieramy: prywatne/lokalne IP i nazwy lokalne (ochrona przed SSRF)."""
    host = host_of(url)
    if host is None:
        return "nieprawidłowy adres"
    if is_ip(host):
        try:
            if not ipaddress.ip_address(host).is_global:
                return "zablokowano adres prywatny lub lokalny"
        except ValueError:
            return "nieprawidłowy adres"
    elif host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan")):
        return "zablokowano adres lokalny"
    return None


def _looks_like_html(data: bytes) -> bool:
    head = data[:2048].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if b"\x00" in head[:512]:
        return False
    return head.startswith((b"<!doctype html", b"<html", b"<?xml")) or any(
        t in head for t in (b"<head", b"<body", b"<script", b"<meta", b"<title", b"<div")
    )


def _read_limited(resp: httpx.Response, max_bytes: int, deadline: float, sniff: bool) -> tuple[bytes | None, bool]:
    """Czyta treść strumieniowo do ``max_bytes``. ``sniff`` — brak typu treści: przerywa, gdy to nie HTML."""
    buf = bytearray()
    truncated = False
    checked = not sniff
    for chunk in resp.iter_bytes(chunk_size=_CHUNK):
        buf += chunk
        if not checked and len(buf) >= 512:
            checked = True
            if not _looks_like_html(bytes(buf)):
                return None, False
        if len(buf) >= max_bytes:
            del buf[max_bytes:]
            truncated = True
            break
        if time.monotonic() > deadline:
            truncated = True
            break
    data = bytes(buf)
    if not checked and not _looks_like_html(data):
        return None, False
    return data, truncated


def _get(client: httpx.Client, url: str, cfg: Any, deadline: float) -> _Hop:
    max_bytes = int(getattr(cfg, "max_bytes", 3_000_000))
    with client.stream("GET", url, headers=request_headers(cfg)) as resp:
        headers = {k.lower(): v for k, v in resp.headers.items()}
        status = resp.status_code
        ctype_full = headers.get("content-type")
        ctype = (ctype_full or "").split(";")[0].strip().lower() or None
        location = headers.get("location")
        if 300 <= status < 400 and location:
            return _Hop(status, headers, ctype_full, location, None)
        body: str | None = None
        truncated = False
        if ctype is None or ctype in TEXT_TYPES:
            data, truncated = _read_limited(resp, max_bytes, deadline, sniff=ctype is None)
            if data is not None:
                body = decode_html_bytes(data, ctype_full)
        return _Hop(status, headers, ctype_full, location, body, truncated)


def _cookie_state(client: httpx.Client) -> tuple:
    try:
        return tuple(sorted((c.name, c.value or "", c.domain, c.path) for c in client.cookies.jar))
    except Exception:  # noqa: BLE001
        return ()


def _join(base: str, target: str) -> str | None:
    try:
        return urljoin(base, target.strip())
    except ValueError:
        return None


# ---------------------------------------------------------------- API


def fetch_chain(
    url: str,
    cfg: Any,
    client: httpx.Client | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
) -> FetchResult:
    """Pobiera ``url`` i podąża za przekierowaniami (HTTP, Refresh, meta refresh, proste JS) do strony końcowej.

    ``cfg`` — NetworkSettings (timeout_s, max_bytes, max_redirects, user_agent). ``client``/``transport`` —
    wstrzyknięcie klienta lub transportu httpx (testy). Nigdy nie rzuca wyjątku."""
    t0 = time.monotonic()
    res = FetchResult()
    timeout = float(getattr(cfg, "timeout_s", 8.0))
    max_redirects = int(getattr(cfg, "max_redirects", 10))
    deadline = t0 + max(timeout * 3.0, timeout + 5.0)

    start = prepare_fetch_url(url) if isinstance(url, str) else None
    if start is None:
        res.error = "nieprawidłowy adres"
        return res

    own_client = client is None
    base_client: httpx.Client | None = None
    insecure: httpx.Client | None = None
    try:
        base_client = client if client is not None else make_client(cfg, transport=transport)
        active = base_client
        current, via = start, "start"
        res.chain = [current]
        visits: dict[str, tuple] = {}
        while True:
            blocked = blocked_reason(current)
            if blocked:
                res.error, res.final_url, res.status = blocked, current, None
                break
            if time.monotonic() > deadline:
                res.error, res.final_url, res.status = "przekroczono łączny czas pobierania", current, None
                break
            key = normalize_url(current) or current
            state = _cookie_state(active)
            if visits.get(key) == state:
                res.error = "pętla przekierowań"
                break
            visits[key] = state

            try:
                hop = _get(active, current, cfg, deadline)
            except Exception as exc:  # noqa: BLE001
                if _is_ssl_error(exc) and not res.tls_invalid:
                    log.info("TLS: nieprawidłowy certyfikat %s — ponawiam bez weryfikacji", host_of(current))
                    res.tls_invalid = True
                    try:
                        insecure = make_client(cfg, verify=False, transport=transport)
                        try:
                            insecure.cookies = active.cookies
                        except Exception:  # noqa: BLE001
                            pass
                        active = insecure
                        hop = _get(active, current, cfg, deadline)
                    except Exception as exc2:  # noqa: BLE001
                        res.error, res.final_url, res.status = error_message(exc2), current, None
                        res.headers, res.content_type = {}, None
                        break
                else:
                    res.error, res.final_url, res.status = error_message(exc), current, None
                    res.headers, res.content_type = {}, None
                    break

            res.final_url, res.status = current, hop.status
            res.headers, res.content_type = hop.headers, hop.content_type
            res.truncated = res.truncated or hop.truncated
            res.hops.append({"url": current, "status": hop.status, "via": via})

            nxt: str | None = None
            nxt_via = ""
            if 300 <= hop.status < 400 and hop.location:
                nxt, nxt_via = _join(current, hop.location), "http"
                if nxt is None:
                    res.error = "nieprawidłowy adres przekierowania"
                    break
            else:
                if hop.body is not None:
                    res.body = hop.body
                    res.pages.append((current, hop.body))
                refresh = hop.headers.get("refresh")
                if refresh:
                    delay, target = parse_refresh(refresh)
                    if target and delay <= REFRESH_MAX_DELAY_S:
                        nxt, nxt_via = _join(current, target), "refresh"
                if nxt is None and hop.body:
                    cr = client_redirect(hop.body, current, max_delay=REFRESH_MAX_DELAY_S)
                    if cr:
                        nxt, nxt_via = cr
            if not nxt:
                break
            if not nxt.lower().startswith(("http://", "https://")):
                res.chain.append(nxt)  # np. tg://… albo intent:// — cel poza siecią WWW, kończymy
                break
            if len(res.chain) - 1 >= max_redirects:
                res.error = f"przekroczono limit przekierowań ({max_redirects})"
                break
            res.chain.append(nxt)
            res.body = None  # treść należy do strony końcowej — następny przeskok ją nadpisze
            current, via = nxt, nxt_via
    except Exception as exc:  # noqa: BLE001 — ostatnia linia obrony: nigdy nie rzucamy
        log.debug("fetch_chain(%s): %s", url, exc, exc_info=True)
        res.error = res.error or error_message(exc)
    finally:
        if own_client and base_client is not None:
            base_client.close()
        if insecure is not None:
            insecure.close()
    res.elapsed_s = round(time.monotonic() - t0, 3)
    return res


def fetch_bytes(
    url: str,
    cfg: Any,
    client: httpx.Client | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    max_bytes: int = FAVICON_MAX_BYTES,
    max_redirects: int = 5,
) -> tuple[bytes | None, str | None, str | None]:
    """Pobiera mały plik (np. favicon) do pamięci — nigdy na dysk. Zwraca (bajty, content-type, błąd).
    Przekierowania śledzone ręcznie (z blokadą adresów prywatnych), certyfikat nie jest weryfikowany
    (liczymy tylko skrót treści)."""
    timeout = float(getattr(cfg, "timeout_s", 8.0))
    deadline = time.monotonic() + max(timeout * 2.0, timeout + 2.0)
    current = prepare_fetch_url(url) if isinstance(url, str) else None
    if current is None:
        return None, None, "nieprawidłowy adres"
    own = client is None
    c: httpx.Client | None = None
    try:
        c = client if client is not None else make_client(cfg, verify=False, transport=transport)
        for _ in range(max_redirects + 1):
            blocked = blocked_reason(current)
            if blocked:
                return None, None, blocked
            with c.stream("GET", current, headers=request_headers(cfg, ACCEPT_IMAGE)) as resp:
                ctype = resp.headers.get("content-type")
                if 300 <= resp.status_code < 400 and resp.headers.get("location"):
                    nxt = _join(current, resp.headers["location"])
                    if not nxt or not nxt.lower().startswith(("http://", "https://")):
                        return None, ctype, "nieprawidłowe przekierowanie"
                    current = nxt
                    continue
                if resp.status_code != 200:
                    return None, ctype, f"HTTP {resp.status_code}"
                buf = bytearray()
                for chunk in resp.iter_bytes(chunk_size=_CHUNK):
                    buf += chunk
                    if len(buf) > max_bytes:
                        return None, ctype, "plik zbyt duży"
                    if time.monotonic() > deadline:
                        return None, ctype, "przekroczono czas"
                return bytes(buf), ctype, None
        return None, None, "przekroczono limit przekierowań"
    except Exception as exc:  # noqa: BLE001
        return None, None, error_message(exc)
    finally:
        if own and c is not None:
            c.close()
