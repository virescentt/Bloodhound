"""Pobieranie stron (httpx.MockTransport — bez sieci): łańcuch przekierowań, meta refresh, JS, pętle, limity,
typy treści, kodowanie znaków, TLS, blokada adresów prywatnych, błędy."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.features.infra.fetch import FetchResult, blocked_reason, fetch_bytes, fetch_chain  # noqa: E402
from slad.settings import NetworkSettings  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "infra"


def html_response(body: str | bytes, status: int = 200, charset: str = "utf-8", **headers: str) -> httpx.Response:
    data = body.encode(charset) if isinstance(body, str) else body
    return httpx.Response(status, headers={"content-type": f"text/html; charset={charset}", **headers}, content=data)


class Server:
    """Prosty „internet” dla MockTransport: (host, ścieżka) → odpowiedź lub funkcja."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.url.host, request.url.path)
        handler = self.routes.get(key, self.routes.get(request.url.host))
        if handler is None:
            return httpx.Response(404, headers={"content-type": "text/html"}, content=b"<h1>404</h1>")
        return handler(request) if callable(handler) else handler

    @property
    def urls(self) -> list[str]:
        return [str(r.url) for r in self.requests]


@pytest.fixture
def cfg() -> NetworkSettings:
    return NetworkSettings(timeout_s=2, max_redirects=10, max_bytes=200_000)


def run(url: str, cfg: NetworkSettings, server: Server) -> FetchResult:
    return fetch_chain(url, cfg, transport=httpx.MockTransport(server))


def test_redirect_chain_meta_refresh_and_js(cfg):
    landing = (FIX / "landing_a1.html").read_text(encoding="utf-8")
    server = Server(
        {
            ("ads.example", "/c"): httpx.Response(
                302, headers={"location": "https://trk.example/click?_lp=1&_token=t"}
            ),
            ("trk.example", "/click"): httpx.Response(301, headers={"location": "/go"}),  # względny Location
            ("trk.example", "/go"): html_response(
                '<html><head><meta http-equiv="refresh" content="0;url=https://pre.example/news"></head></html>'
            ),
            ("pre.example", "/news"): html_response(
                "<html><head><title>pre</title><script>fbq('init', '111122223333444');"
                "window.location.replace('https://land.example/lp')</script></head></html>"
            ),
            ("land.example", "/lp"): html_response(landing),
        }
    )
    r = run("https://ads.example/c?id=1#frag", cfg, server)
    assert r.error is None
    assert r.chain == [
        "https://ads.example/c?id=1",
        "https://trk.example/click?_lp=1&_token=t",
        "https://trk.example/go",
        "https://pre.example/news",
        "https://land.example/lp",
    ]
    assert r.final_url == "https://land.example/lp"
    assert r.status == 200
    assert "Jan Kowalski" in (r.body or "")
    assert [h["via"] for h in r.hops] == ["start", "http", "http", "meta", "js"]
    assert [u for u, _ in r.pages] == ["https://trk.example/go", "https://pre.example/news", "https://land.example/lp"]
    assert r.headers["content-type"].startswith("text/html")
    # nagłówki przeglądarki: User-Agent z konfiguracji i polski Accept-Language
    first = server.requests[0]
    assert first.headers["user-agent"] == cfg.user_agent
    assert first.headers["accept-language"].startswith("pl")
    assert "#frag" not in server.urls[0]


def test_refresh_header_and_non_web_target(cfg):
    server = Server(
        {
            ("a.example", "/"): httpx.Response(200, headers={"refresh": "0; url=/b", "content-type": "text/html"}),
            ("a.example", "/b"): httpx.Response(302, headers={"location": "tg://resolve?domain=kanal_oszusta"}),
        }
    )
    r = run("http://a.example/", cfg, server)
    assert r.chain == ["http://a.example/", "http://a.example/b", "tg://resolve?domain=kanal_oszusta"]
    assert r.final_url == "http://a.example/b" and r.status == 302 and r.error is None


def test_redirect_loop_detected(cfg):
    server = Server(
        {
            "loop-a.example": httpx.Response(302, headers={"location": "https://loop-b.example/"}),
            "loop-b.example": httpx.Response(302, headers={"location": "https://loop-a.example/"}),
        }
    )
    r = run("https://loop-a.example/", cfg, server)
    assert r.error == "pętla przekierowań"
    assert r.chain == ["https://loop-a.example/", "https://loop-b.example/", "https://loop-a.example/"]
    assert len(server.requests) == 2
    assert r.final_url == "https://loop-b.example/" and r.status == 302


def test_revisit_after_cookie_is_not_a_loop(cfg):
    def gate(request: httpx.Request) -> httpx.Response:
        if "ok=1" in request.headers.get("cookie", ""):
            return html_response("<html><title>po weryfikacji</title><body>treść</body></html>")
        return httpx.Response(302, headers={"location": "https://check.scam-site.com/set"})

    server = Server(
        {
            "gate.scam-site.com": gate,
            ("check.scam-site.com", "/set"): httpx.Response(
                302,
                headers={
                    "location": "https://gate.scam-site.com/",
                    "set-cookie": "ok=1; Domain=.scam-site.com; Path=/",
                },
            ),
        }
    )
    r = run("https://gate.scam-site.com/", cfg, server)
    assert r.error is None and r.status == 200
    assert "po weryfikacji" in r.body
    assert r.chain == ["https://gate.scam-site.com/", "https://check.scam-site.com/set", "https://gate.scam-site.com/"]


def test_max_redirects(cfg):
    def hop(request: httpx.Request) -> httpx.Response:
        n = int(request.url.path.strip("/") or 0)
        return httpx.Response(302, headers={"location": f"/{n + 1}"})

    server = Server({"chain.example": hop})
    r = run("http://chain.example/0", NetworkSettings(timeout_s=2, max_redirects=3), server)
    assert r.error == "przekroczono limit przekierowań (3)"
    assert r.chain == [
        "http://chain.example/0",
        "http://chain.example/1",
        "http://chain.example/2",
        "http://chain.example/3",
    ]
    assert len(server.requests) == 4


def test_max_bytes_truncates_stream(cfg):
    big = "<html><body>" + "<p>tekst</p>" * 50_000 + "</body></html>"
    server = Server({"big.example": html_response(big)})
    r = run("big.example", NetworkSettings(timeout_s=2, max_bytes=10_000), server)
    assert r.status == 200 and r.truncated
    assert r.body is not None and len(r.body.encode()) <= 10_000
    assert r.final_url == "http://big.example/"


def test_binary_content_is_not_read(cfg):
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 5000
    server = Server(
        {
            "img.example": httpx.Response(200, headers={"content-type": "image/png"}, content=png),
            "exe.example": httpx.Response(
                200, headers={"content-type": "application/octet-stream"}, content=b"MZ" * 100
            ),
            "noct.example": httpx.Response(200, content=b"\x00\x01\x02" * 400),  # brak typu, binarne
            "noct-html.example": httpx.Response(200, content=b"<!DOCTYPE html><html><title>t</title></html>"),
        }
    )
    for host in ("img.example", "exe.example", "noct.example"):
        r = run(f"https://{host}/x", cfg, server)
        assert r.status == 200 and r.body is None and r.pages == [], host
    r = run("https://noct-html.example/", cfg, server)
    assert r.body and "<title>t</title>" in r.body  # HTML rozpoznany po treści


def test_charset_from_header_and_meta(cfg):
    text = "Zażółć gęślą jaźń"
    server = Server(
        {
            "h.example": html_response(f"<html><body>{text}</body></html>", charset="windows-1250"),
            "m.example": httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=f'<html><head><meta charset="iso-8859-2"></head><body>{text}</body></html>'.encode(
                    "iso-8859-2"
                ),
            ),
        }
    )
    assert text in run("https://h.example/", cfg, server).body
    assert text in run("https://m.example/", cfg, server).body


def test_ssl_error_retries_without_verification(cfg):
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed", request=request)
        return html_response("<html><body>strona z błędnym certyfikatem</body></html>")

    r = run("https://badcert.example/", cfg, Server({"badcert.example": flaky}))
    assert r.tls_invalid is True
    assert r.status == 200 and r.error is None
    assert "błędnym certyfikatem" in r.body
    assert calls["n"] == 2


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (httpx.ReadTimeout("timed out"), "przekroczono czas"),
        (httpx.ConnectTimeout("timed out"), "przekroczono czas"),
        (
            httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known"),
            "nie można rozwiązać nazwy domeny",
        ),
        (httpx.ConnectError("[Errno 61] Connection refused"), "połączenie odrzucone"),
        (httpx.RemoteProtocolError("Server disconnected"), "błąd protokołu HTTP"),
        (RuntimeError("cokolwiek"), "błąd (RuntimeError)"),
    ],
)
def test_errors_never_raise(cfg, exc, message):
    def boom(request: httpx.Request) -> httpx.Response:
        raise exc

    r = run("https://dead.example/x", cfg, Server({"dead.example": boom}))
    assert r.error == message
    assert r.status is None
    assert r.final_url == "https://dead.example/x"
    assert r.chain == ["https://dead.example/x"]


def test_error_mid_chain_keeps_partial_chain(cfg):
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")

    server = Server(
        {"a.example": httpx.Response(302, headers={"location": "https://gone.example/"}), "gone.example": boom}
    )
    r = run("https://a.example/", cfg, server)
    assert r.chain == ["https://a.example/", "https://gone.example/"]
    assert r.final_url == "https://gone.example/" and r.status is None
    assert r.error == "nie można rozwiązać nazwy domeny"


def test_private_addresses_blocked_without_request(cfg):
    server = Server({})
    for url in (
        "http://127.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://localhost:8501/",
        "http://[::1]/",
    ):
        r = run(url, cfg, server)
        assert r.error in ("zablokowano adres prywatny lub lokalny", "nieprawidłowy adres"), url
    assert server.requests == []
    # przekierowanie do sieci wewnętrznej też jest blokowane
    server = Server({"evil.example": httpx.Response(302, headers={"location": "http://10.0.0.5/internal"})})
    r = run("https://evil.example/", cfg, server)
    assert r.error == "zablokowano adres prywatny lub lokalny"
    assert server.urls == ["https://evil.example/"]
    assert blocked_reason("https://scam.pl/") is None


def test_invalid_input(cfg):
    for bad in (None, "", "brak", "javascript:alert(1)", "mailto:a@b.pl", 12345):
        r = fetch_chain(bad, cfg, transport=httpx.MockTransport(Server({})))  # type: ignore[arg-type]
        assert r.error == "nieprawidłowy adres" and r.chain == []


def test_injected_client_is_not_closed(cfg):
    server = Server({"a.example": html_response("<html><body>ok</body></html>")})
    client = httpx.Client(transport=httpx.MockTransport(server), follow_redirects=False)
    r1 = fetch_chain("https://a.example/", cfg, client=client)
    r2 = fetch_chain("https://a.example/", cfg, client=client)
    assert r1.status == r2.status == 200
    assert not client.is_closed
    client.close()


def test_fetch_bytes_favicon(cfg):
    png = b"\x89PNG\r\n\x1a\n" + b"\x01" * 100
    server = Server(
        {
            ("s.example", "/favicon.ico"): httpx.Response(301, headers={"location": "/static/fav.png"}),
            ("s.example", "/static/fav.png"): httpx.Response(200, headers={"content-type": "image/png"}, content=png),
            ("big.example", "/favicon.ico"): httpx.Response(200, content=b"\x00" * 2_000_000),
            ("r.example", "/favicon.ico"): httpx.Response(302, headers={"location": "http://192.168.0.1/x.ico"}),
        }
    )
    t = httpx.MockTransport(server)
    assert fetch_bytes("https://s.example/favicon.ico", cfg, transport=t) == (png, "image/png", None)
    assert fetch_bytes("https://none.example/favicon.ico", cfg, transport=t)[2] == "HTTP 404"
    assert fetch_bytes("https://big.example/favicon.ico", cfg, transport=t)[2] == "plik zbyt duży"
    assert fetch_bytes("https://r.example/favicon.ico", cfg, transport=t)[2] == "zablokowano adres prywatny lub lokalny"
    assert fetch_bytes("brak", cfg, transport=t)[2] == "nieprawidłowy adres"
