"""Zapytania o infrastrukturę sieciową: DNS (A, AAAA, CNAME, NS, MX), ASN adresu IP (Team Cymru przez DNS),
RDAP domeny (rejestrator, data rejestracji, serwery nazw, status) i RDAP adresu IP (nazwa sieci).

Każda funkcja ma limit czasu, NIGDY nie rzuca wyjątku i zwraca częściowy wynik + błędy (krótko, po polsku).
Klient DNS (``resolver``) i HTTP (``client``) można wstrzyknąć — testy działają bez sieci.
"""

from __future__ import annotations

import ipaddress
import logging
import threading
import time
from datetime import UTC, datetime
from typing import Any

import dns.exception
import dns.resolver
import httpx

from slad.features.infra.domains import clean_host, host_of, is_ip, registered_domain, unique

log = logging.getLogger(__name__)

RDAP_BOOTSTRAP = "https://rdap.org"
FALLBACK_NAMESERVERS = ["1.1.1.1", "8.8.8.8", "9.9.9.9"]
_UA = "SladFejka/0.1 (+badania kampanii oszustw; HackNation 2026)"

_AS_NAMES: dict[str, str] = {}  # pamięć nazw systemów autonomicznych w procesie (AS13335 → CLOUDFLARENET)
_AS_LOCK = threading.Lock()


# ---------------------------------------------------------------- DNS


def make_resolver(timeout: float = 5.0) -> Any:
    """Resolver dnspython z konfiguracji systemu; bez niej (np. kontener) — publiczne serwery DNS."""
    try:
        r = dns.resolver.Resolver()
        if not r.nameservers:
            raise dns.resolver.NoResolverConfiguration("brak serwerów")
    except Exception:  # noqa: BLE001
        r = dns.resolver.Resolver(configure=False)
        r.nameservers = list(FALLBACK_NAMESERVERS)
    r.timeout = float(timeout)
    r.lifetime = float(timeout)
    return r


class _BudgetExceeded(Exception):
    """Wyczerpany łączny limit czasu zapytań DNS dla jednego hosta."""


def _dns_error(exc: BaseException) -> str:
    if isinstance(exc, dns.resolver.NXDOMAIN):
        return "dns: domena nie istnieje"
    if isinstance(exc, dns.exception.Timeout):
        return "dns: przekroczono czas"
    if isinstance(exc, dns.resolver.NoNameservers):
        return "dns: serwery nazw nie odpowiadają"
    if isinstance(exc, _BudgetExceeded):
        return "dns: przekroczono czas"
    return f"dns: błąd ({type(exc).__name__})"


def _name(value: Any) -> str:
    return str(value).rstrip(".").lower()


def _records(resolver: Any, name: str, rdtype: str, timeout: float) -> list[Any]:
    """Rekordy danego typu; brak odpowiedzi (NoAnswer) → []; inne błędy → wyjątek."""
    try:
        answer = resolver.resolve(name, rdtype, lifetime=timeout)
    except dns.resolver.NoAnswer:
        return []
    return list(answer)


def dns_info(host: str, timeout: float = 5.0, *, resolver: Any = None) -> dict[str, Any]:
    """Rekordy DNS hosta: ``{a, aaaa, cname, ns, mx, errors}``. NS i MX — dla domeny zarejestrowanej."""
    out: dict[str, Any] = {"a": [], "aaaa": [], "cname": [], "ns": [], "mx": [], "errors": []}
    h = host_of(host) if host else None
    if not h:
        out["errors"].append("dns: nieprawidłowa nazwa hosta")
        return out
    if is_ip(h):
        out["a" if ipaddress.ip_address(h).version == 4 else "aaaa"] = [h]
        return out
    try:
        res = resolver if resolver is not None else make_resolver(timeout)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"dns: brak resolwera ({type(e).__name__})")
        return out
    deadline = time.monotonic() + max(2.0 * timeout, 2.0)

    def query(name: str, rdtype: str) -> list[Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _BudgetExceeded()
        return _records(res, name, rdtype, min(float(timeout), remaining))

    def add_error(exc: BaseException) -> None:
        msg = _dns_error(exc)
        if msg not in out["errors"]:
            out["errors"].append(msg)

    host_missing = False
    for rdtype, key in (("A", "a"), ("AAAA", "aaaa"), ("CNAME", "cname")):
        if host_missing:
            break
        try:
            recs = query(h, rdtype)
            if rdtype in ("A", "AAAA"):
                out[key] = unique(str(getattr(r, "address", None) or r.to_text()) for r in recs)
            else:
                out[key] = unique(_name(getattr(r, "target", None) or r.to_text()) for r in recs)
        except Exception as e:  # noqa: BLE001
            add_error(e)
            if isinstance(e, (dns.resolver.NXDOMAIN, _BudgetExceeded)):
                host_missing = True

    zone = registered_domain(h) or h
    for rdtype in ("NS", "MX"):
        try:
            recs = query(zone, rdtype)
            if rdtype == "NS":
                out["ns"] = sorted(unique(_name(getattr(r, "target", None) or r.to_text()) for r in recs))
            else:
                pairs = []
                for r in recs:
                    exch = getattr(r, "exchange", None)
                    pref = getattr(r, "preference", 0)
                    if exch is None:  # tekst „10 mx.example.com.”
                        parts = r.to_text().split()
                        pref, exch = (int(parts[0]), parts[-1]) if len(parts) >= 2 else (0, parts[-1])
                    pairs.append((int(pref or 0), _name(exch)))
                out["mx"] = unique(name for _, name in sorted(pairs) if name)
        except Exception as e:  # noqa: BLE001
            add_error(e)
    return out


# ---------------------------------------------------------------- ASN (Team Cymru)


def _txt(resolver: Any, name: str, timeout: float) -> list[str]:
    out = []
    for r in _records(resolver, name, "TXT", timeout):
        strings = getattr(r, "strings", None)
        if strings:
            out.append(
                b"".join(s if isinstance(s, bytes) else str(s).encode() for s in strings).decode("utf-8", "replace")
            )
        else:
            out.append(r.to_text().strip('"'))
    return out


def parse_cymru_origin(records: list[str]) -> str | None:
    """``"13335 | 104.16.0.0/13 | US | arin | 2014-03-28"`` → ``"13335"`` (najdłuższy prefiks wygrywa)."""
    best: tuple[int, str] | None = None
    for txt in records:
        fields = [f.strip() for f in txt.strip().strip('"').split("|")]
        if not fields or not fields[0]:
            continue
        asn = fields[0].split()[0]
        if not asn.isdigit():
            continue
        plen = 0
        if len(fields) > 1 and "/" in fields[1]:
            try:
                plen = int(fields[1].rsplit("/", 1)[1])
            except ValueError:
                plen = 0
        if best is None or plen > best[0]:
            best = (plen, asn)
    return best[1] if best else None


def parse_cymru_asname(records: list[str]) -> str | None:
    """``"13335 | US | arin | 2010-07-14 | CLOUDFLARENET - Cloudflare, Inc., US"`` → ``"CLOUDFLARENET"``."""
    for txt in records:
        fields = [f.strip() for f in txt.strip().strip('"').split("|")]
        if len(fields) < 5 or not fields[-1]:
            continue
        name = fields[-1]
        if len(name) > 4 and name[-4:-2] == ", " and name[-2:].isalpha() and name[-2:].isupper():
            name = name[:-4]  # „…, US”
        name = name.split(" - ", 1)[0].strip()
        if name:
            return name[:80]
    return None


def asn_lookup(ip: str, timeout: float = 5.0, *, resolver: Any = None) -> tuple[str | None, str | None]:
    """(``"AS13335 CLOUDFLARENET"`` | None, błąd | None). Adres niepubliczny → (None, None)."""
    try:
        addr = ipaddress.ip_address(str(ip).strip())
    except ValueError:
        return None, "asn: nieprawidłowy adres IP"
    if not addr.is_global:
        return None, None
    if addr.version == 4:
        qname = ".".join(reversed(str(addr).split("."))) + ".origin.asn.cymru.com"
    else:
        qname = ".".join(reversed(addr.exploded.replace(":", ""))) + ".origin6.asn.cymru.com"
    try:
        res = resolver if resolver is not None else make_resolver(timeout)
        asn = parse_cymru_origin(_txt(res, qname, timeout))
        if not asn:
            return None, "asn: brak danych"
        with _AS_LOCK:
            name = _AS_NAMES.get(asn)
        if name is None:
            try:
                name = parse_cymru_asname(_txt(res, f"AS{asn}.asn.cymru.com", timeout)) or ""
            except Exception as e:  # noqa: BLE001 — sam numer AS też jest wartościowy
                log.debug("nazwa AS%s: %s", asn, e)
                name = ""
            with _AS_LOCK:
                _AS_NAMES[asn] = name
        return (f"AS{asn} {name}" if name else f"AS{asn}"), None
    except Exception as e:  # noqa: BLE001
        msg = _dns_error(e).replace("dns:", "asn:", 1)
        return None, msg


def asn_for_ip(ip: str, timeout: float = 5.0, *, resolver: Any = None) -> str | None:
    """``"AS<n> <NAZWA>"`` dla adresu IP (Team Cymru, DNS) albo None."""
    return asn_lookup(ip, timeout, resolver=resolver)[0]


# ---------------------------------------------------------------- RDAP


def _get_json(url: str, timeout: float, client: httpx.Client | None) -> tuple[Any, str | None]:
    own = client is None
    c = client
    try:
        if c is None:
            c = httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": _UA})
        resp = c.get(
            url,
            headers={"Accept": "application/rdap+json, application/json;q=0.9"},
            follow_redirects=True,
            timeout=timeout,
        )
        if resp.status_code == 404:
            return None, "brak danych (HTTP 404)"
        if resp.status_code == 429:
            return None, "limit zapytań (HTTP 429)"
        if resp.status_code >= 400:
            return None, f"HTTP {resp.status_code}"
        if len(resp.content) > 5_000_000:
            return None, "zbyt duża odpowiedź"
        return resp.json(), None
    except httpx.TimeoutException:
        return None, "przekroczono czas"
    except ValueError:
        return None, "nieprawidłowa odpowiedź"
    except httpx.HTTPError as e:
        return None, f"błąd połączenia ({type(e).__name__})"
    except Exception as e:  # noqa: BLE001
        return None, f"błąd ({type(e).__name__})"
    finally:
        if own and c is not None:
            c.close()


def to_iso(value: Any) -> str | None:
    """Data z RDAP/WHOIS → ISO 8601 w UTC (``2025-10-01T12:00:00+00:00``); nieczytelna → None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00") if s.endswith("Z") else s)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _vcard_field(entity: dict[str, Any], *names: str) -> str | None:
    vc = entity.get("vcardArray")
    if not (isinstance(vc, list) and len(vc) >= 2 and isinstance(vc[1], list)):
        return None
    for name in names:
        for item in vc[1]:
            if isinstance(item, list) and len(item) >= 4 and str(item[0]).lower() == name:
                val = item[3]
                if isinstance(val, list):
                    val = " ".join(str(x) for x in val if x)
                if val and str(val).strip():
                    return str(val).strip()
    return None


def _walk_entities(entities: Any, depth: int = 0):
    if not isinstance(entities, list) or depth > 4:
        return
    for ent in entities:
        if isinstance(ent, dict):
            yield ent
            yield from _walk_entities(ent.get("entities"), depth + 1)


def parse_rdap_domain(data: Any) -> dict[str, Any]:
    """Odpowiedź RDAP domeny → ``{registrar, created, expires, updated, nameservers, status}``."""
    out: dict[str, Any] = {
        "registrar": None,
        "created": None,
        "expires": None,
        "updated": None,
        "nameservers": [],
        "status": [],
    }
    if not isinstance(data, dict):
        return out
    for ent in _walk_entities(data.get("entities")):
        roles = [str(r).lower() for r in ent.get("roles") or []]
        if "registrar" in roles:
            name = _vcard_field(ent, "fn", "org") or ent.get("handle")
            out["registrar"] = str(name).strip()[:120] if name else None
            break
    for ev in data.get("events") or []:
        if not isinstance(ev, dict):
            continue
        action = str(ev.get("eventAction", "")).lower()
        key = {"registration": "created", "expiration": "expires", "last changed": "updated"}.get(action)
        if key and out[key] is None:
            out[key] = to_iso(ev.get("eventDate"))
    ns = []
    for item in data.get("nameservers") or []:
        if isinstance(item, dict):
            name = clean_host(str(item.get("ldhName") or item.get("unicodeName") or ""))
            if name:
                ns.append(name)
    out["nameservers"] = sorted(unique(ns))
    out["status"] = [str(s) for s in data.get("status") or [] if s]
    return out


def rdap_domain(domain: str, timeout: float = 8.0, *, client: httpx.Client | None = None) -> dict[str, Any]:
    """RDAP domeny (przez rdap.org → serwer rejestru): ``{registrar, created (ISO), expires, updated,
    nameservers, status, error}``."""
    d = registered_domain(domain) or clean_host(domain)
    if not d:
        return {**parse_rdap_domain(None), "error": "rdap: nieprawidłowa domena"}
    data, err = _get_json(f"{RDAP_BOOTSTRAP}/domain/{d}", timeout, client)
    if err:
        return {**parse_rdap_domain(None), "error": f"rdap: {err}"}
    return {**parse_rdap_domain(data), "error": None}


def parse_rdap_ip(data: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"name": None, "handle": None, "country": None, "cidr": None, "asn": None}
    if not isinstance(data, dict):
        return out
    out["name"] = (str(data.get("name")).strip() or None) if data.get("name") else None
    out["handle"] = data.get("handle")
    out["country"] = data.get("country")
    cidrs = data.get("cidr0_cidrs")
    if isinstance(cidrs, list) and cidrs and isinstance(cidrs[0], dict):
        c = cidrs[0]
        prefix = c.get("v4prefix") or c.get("v6prefix")
        if prefix and c.get("length") is not None:
            out["cidr"] = f"{prefix}/{c['length']}"
    if not out["cidr"] and data.get("startAddress") and data.get("endAddress"):
        out["cidr"] = f"{data['startAddress']} - {data['endAddress']}"
    origin = data.get("arin_originas0_originautnums")
    if isinstance(origin, list) and origin:
        out["asn"] = f"AS{origin[0]}"
    return out


def rdap_ip(ip: str, timeout: float = 8.0, *, client: httpx.Client | None = None) -> dict[str, Any]:
    """RDAP adresu IP — nazwa sieci (zapas, gdy Team Cymru nie odpowiada): ``{name, handle, country, cidr, asn,
    error}``."""
    try:
        addr = ipaddress.ip_address(str(ip).strip())
    except ValueError:
        return {**parse_rdap_ip(None), "error": "rdap: nieprawidłowy adres IP"}
    data, err = _get_json(f"{RDAP_BOOTSTRAP}/ip/{addr}", timeout, client)
    if err:
        return {**parse_rdap_ip(None), "error": f"rdap: {err}"}
    return {**parse_rdap_ip(data), "error": None}
