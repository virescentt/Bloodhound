"""Adapter pliku HAR (HTTP Archive, standard HAR 1.2 z Chrome/Playwright/Firefox) → meta["infra"].

HAR zapisuje wszystkie żądania strony: łańcuch przekierowań, hosty i domeny, adresy IP serwerów, skrypty,
identyfikatory trackerów oraz (w Chrome) dane certyfikatu TLS. Wynik to słownik w kształcie, który rozumie
``slad.features.infra`` (klucze jak w ``InfraFeatures``), więc trafia do ``Material.meta["infra"]``.

Plik traktujemy jako dane niezaufane: z HAR tylko czytamy napisy, nic nie jest pobierane ani uruchamiane.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

MAX_HAR_BYTES = 80_000_000  # większe pliki odrzucamy zamiast ładować do pamięci
MAX_LIST = 200  # limit długości list (hosty, domeny, IP) — strony reklamowe nie mają tysięcy hostów
MAX_CHAIN = 30  # długość łańcucha przekierowań, po której przerywamy (pętle)
MAX_SANS = 50

_REDIRECT_CODES = range(300, 400)

# identyfikatory trackerów widoczne w adresach żądań: (klucz meta["infra"], wzorzec, nazwa grupy z identyfikatorem)
_TRACKER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("tracker", re.compile(r"googletagmanager\.com/(?:gtag/js|gtm\.js)\?(?:[^#]*&)?id=((?:G|GTM|UA)-[A-Za-z0-9-]+)")),
    ("tracker", re.compile(r"google-analytics\.com/(?:g/)?collect\?(?:[^#]*&)?tid=((?:UA|G)-[A-Za-z0-9-]+)")),
    ("fb_pixel", re.compile(r"facebook\.com/tr\?(?:[^#]*&)?id=(\d{5,20})")),
    ("ym", re.compile(r"mc\.yandex\.ru/watch/(\d{5,12})")),
    ("tiktok_pixel", re.compile(r"analytics\.tiktok\.com/i18n/pixel/events\.js\?(?:[^#]*&)?sdkid=([A-Z0-9]{5,40})")),
)


def _str(v: Any) -> str | None:
    return v.strip() if isinstance(v, str) and v.strip() else None


def _dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _entries(data: Any) -> list[dict[str, Any]]:
    log = _dict(data).get("log")
    if not isinstance(log, dict) or not isinstance(log.get("entries"), list):
        raise ValueError("to nie jest plik HAR (brak log.entries)")
    return [e for e in log["entries"] if isinstance(e, dict)]


def _url(e: dict[str, Any]) -> str | None:
    return _str(_dict(e.get("request")).get("url"))


def _resp(e: dict[str, Any]) -> dict[str, Any]:
    return _dict(e.get("response"))


def _status(e: dict[str, Any]) -> int | None:
    s = _resp(e).get("status")
    return s if isinstance(s, int) and not isinstance(s, bool) else None


def _rtype(e: dict[str, Any]) -> str:
    return str(e.get("_resourceType") or "").lower()


def _mime(e: dict[str, Any]) -> str:
    return (_str(_dict(_resp(e).get("content")).get("mimeType")) or "").lower()


def _host(url: str | None) -> str | None:
    if not url:
        return None
    try:
        h = urlsplit(url).hostname
    except ValueError:
        return None
    return h.lower().rstrip(".") if h else None


def _main_chain(entries: list[dict[str, Any]]) -> tuple[list[str], int | None, str | None]:
    """Łańcuch przekierowań głównego dokumentu: (adresy po kolei, status końcowej odpowiedzi, adres końcowy)."""
    docs = [e for e in entries if _rtype(e) == "document"] or entries
    chain: list[str] = []
    cur: str | None = None
    status: int | None = None
    for e in docs:
        u = _url(e)
        if cur is None:
            if not u:
                continue
            cur = u
            chain.append(u)
        if u != cur:
            continue
        status = _status(e)
        loc = _str(_resp(e).get("redirectURL"))
        if status in _REDIRECT_CODES and loc and len(chain) < MAX_CHAIN:
            cur = urljoin(cur, loc)
            chain.append(cur)
        else:
            break
    return list(dict.fromkeys(chain)), status, cur


def _tls_details(entries: list[dict[str, Any]], final_url: str | None) -> tuple[str | None, list[str]]:
    """Wystawca i nazwy alternatywne certyfikatu (Chrome zapisuje je w _securityDetails dokumentu)."""
    for e in entries:
        if final_url and _url(e) != final_url:
            continue
        sec = _dict(e.get("_securityDetails"))
        if sec:
            issuer = _str(sec.get("issuer"))
            raw = sec.get("sanList")
            raw = raw if isinstance(raw, list) else []
            sans = [s for s in (_str(x) for x in raw[:MAX_SANS]) if s]
            return issuer, sans
    return None, []


def har_to_seed(data: Any) -> dict[str, Any]:
    """Słownik HAR (już wczytany JSON) → meta["infra"]. Pusta lista/None oznacza „brak danych”."""
    from slad.features.infra.domains import registered_domain

    entries = _entries(data)
    chain, status, final_url = _main_chain(entries)

    hosts: list[str] = []
    ips: list[str] = []
    script_domains: list[str] = []
    trackers: dict[str, list[str]] = {}
    no_response = 0
    for e in entries:
        u = _url(e)
        h = _host(u)
        if h and h not in hosts:
            hosts.append(h)
        ip = _str(e.get("serverIPAddress"))
        if ip and ip not in ips:
            ips.append(ip)
        if _rtype(e) == "script" or "javascript" in _mime(e):
            d = registered_domain(h)
            if d and d not in script_domains:
                script_domains.append(d)
        if _status(e) == 0:
            no_response += 1
        for key, pattern in _TRACKER_PATTERNS:
            for m in pattern.finditer(u or ""):
                ident = m.group(1)
                hint = _tracker_kind(ident, key)
                bucket = trackers.setdefault(hint, [])
                if ident not in bucket:
                    bucket.append(ident)

    domains: list[str] = []
    for h in hosts:
        d = registered_domain(h)
        if d and d not in domains:
            domains.append(d)

    pages = _dict(data.get("log")).get("pages") if isinstance(data, dict) else None
    title = None
    if isinstance(pages, list) and pages and isinstance(pages[0], dict):
        title = _str(pages[0].get("title"))
    tls_issuer, tls_sans = _tls_details(entries, final_url)

    errors: list[str] = []
    if not entries:
        errors.append("HAR: brak żądań w pliku")
    if no_response:
        errors.append(f"HAR: {no_response} żądań bez odpowiedzi serwera (błąd sieci lub blokada)")

    seed: dict[str, Any] = {
        "redirect_chain": chain,
        "final_url": final_url,
        "http_status": status,
        "hosts": hosts[:MAX_LIST],
        "domains": domains[:MAX_LIST],
        "script_domains": script_domains[:MAX_LIST],
        "ips": ips[:MAX_LIST],
        "title": title,
        "tls_issuer": tls_issuer,
        "tls_sans": tls_sans,
        "errors": errors,
    }
    seed.update({k: v[:MAX_LIST] for k, v in trackers.items()})
    return {k: v for k, v in seed.items() if v not in (None, [], "")}


def _tracker_kind(ident: str, key: str) -> str:
    """Rodzaj identyfikatora z prefiksu (G-, GTM-, UA-) — klucze zgodne z infra._SEED_TRACKER_KEYS."""
    if key != "tracker":
        return key
    if ident.startswith("GTM-"):
        return "gtm"
    if ident.startswith("G-"):
        return "gtag"
    return "ga"


def read_har(path: Path | str) -> dict[str, Any]:
    """Wczytuje plik HAR i zwraca meta["infra"]. Rzuca ValueError przy uszkodzonym pliku lub zbyt dużym."""
    p = Path(path)
    if p.stat().st_size > MAX_HAR_BYTES:
        raise ValueError(f"plik HAR za duży ({p.stat().st_size} B > {MAX_HAR_BYTES} B)")
    try:
        data = json.loads(p.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"plik HAR nie jest poprawnym JSON-em: {e}") from e
    return har_to_seed(data)
