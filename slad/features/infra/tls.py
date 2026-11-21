"""Certyfikat TLS serwera (wystawca, podmiot, odcisk SHA-256, nazwy SAN, ważność) i Certificate Transparency
(crt.sh — inne nazwy z certyfikatów domeny).

Certyfikat pobieramy BEZ weryfikacji (``CERT_NONE``), żeby odczytać go także z serwerów z nieprawidłowym
certyfikatem — to częste na stronach oszustów. Funkcje nigdy nie rzucają wyjątków.
"""

from __future__ import annotations

import hashlib
import logging
import socket
import ssl
from typing import Any

import httpx

from slad.features.infra.domains import clean_host, is_ip, registered_domain, unique

log = logging.getLogger(__name__)

CRTSH_URL = "https://crt.sh/"
CRTSH_CAP = 200
SANS_CAP = 100


def _empty() -> dict[str, Any]:
    return {
        "issuer": None,
        "subject": None,
        "fingerprint": None,
        "sans": [],
        "not_before": None,
        "not_after": None,
        "error": None,
    }


def _name_attr(name: Any, oid: Any) -> str | None:
    try:
        attrs = name.get_attributes_for_oid(oid)
    except Exception:  # noqa: BLE001
        return None
    for a in attrs:
        v = str(a.value).strip()
        if v:
            return v
    return None


def cert_info(der: bytes) -> dict[str, Any]:
    """Certyfikat DER → ``{issuer, subject, fingerprint, sans, not_before, not_after, error}``.
    ``issuer`` = „Organizacja (CN)”, np. „Let's Encrypt (R11)”; ``fingerprint`` = sha256(DER) hex."""
    out = _empty()
    if not der:
        out["error"] = "tls: brak certyfikatu"
        return out
    out["fingerprint"] = hashlib.sha256(der).hexdigest()
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID

        cert = x509.load_der_x509_certificate(der)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"tls: nie można odczytać certyfikatu ({type(e).__name__})"
        return out
    try:
        org = _name_attr(cert.issuer, NameOID.ORGANIZATION_NAME)
        cn = _name_attr(cert.issuer, NameOID.COMMON_NAME)
        if org and cn and cn != org:
            out["issuer"] = f"{org} ({cn})"
        else:
            out["issuer"] = org or cn or cert.issuer.rfc4514_string() or None
        out["subject"] = (
            _name_attr(cert.subject, NameOID.COMMON_NAME)
            or _name_attr(cert.subject, NameOID.ORGANIZATION_NAME)
            or cert.subject.rfc4514_string()
            or None
        )
    except Exception as e:  # noqa: BLE001
        log.debug("tls: nazwy certyfikatu: %s", e)
    try:
        ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        names = [clean_host(n.lstrip("*.")) for n in ext.value.get_values_for_type(x509.DNSName)]
        out["sans"] = unique(n for n in names if n)[:SANS_CAP]
    except x509.ExtensionNotFound:
        pass
    except Exception as e:  # noqa: BLE001
        log.debug("tls: SAN: %s", e)
    try:
        out["not_before"] = cert.not_valid_before_utc.isoformat()
        out["not_after"] = cert.not_valid_after_utc.isoformat()
    except Exception as e:  # noqa: BLE001
        log.debug("tls: daty: %s", e)
    return out


def tls_info(host: str, port: int = 443, timeout: float = 8.0) -> dict[str, Any]:
    """Certyfikat serwera ``host:port`` (bez weryfikacji). Błędy → pole ``error`` (po polsku)."""
    out = _empty()
    h = clean_host(host)
    if not h:
        out["error"] = "tls: nieprawidłowa nazwa hosta"
        return out
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:  # tylko odczyt certyfikatu — dopuszczamy stare protokoły i słabe klucze (częste u oszustów)
        ctx.minimum_version = ssl.TLSVersion.MINIMUM_SUPPORTED
        ctx.set_ciphers("DEFAULT:@SECLEVEL=0")
    except (ssl.SSLError, ValueError, AttributeError):
        pass
    try:
        with socket.create_connection((h, int(port)), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=None if is_ip(h) else h) as ssock:
                der = ssock.getpeercert(binary_form=True)
    except TimeoutError:
        out["error"] = "tls: przekroczono czas"
        return out
    except ssl.SSLError as e:
        out["error"] = f"tls: błąd uzgadniania TLS ({e.reason or type(e).__name__})"
        return out
    except socket.gaierror:
        out["error"] = "tls: nie można rozwiązać nazwy domeny"
        return out
    except OSError as e:
        out["error"] = f"tls: błąd połączenia ({type(e).__name__})"
        return out
    except Exception as e:  # noqa: BLE001
        out["error"] = f"tls: błąd ({type(e).__name__})"
        return out
    return cert_info(der or b"")


def parse_crtsh(data: Any, cap: int = CRTSH_CAP) -> list[str]:
    """Odpowiedź JSON crt.sh → unikalne nazwy hostów (bez ``*.``), najwyżej ``cap``."""
    names: list[str] = []
    if not isinstance(data, list):
        return names
    for row in data:
        if not isinstance(row, dict):
            continue
        for field in ("common_name", "name_value"):
            for raw in str(row.get(field) or "").split("\n"):
                n = clean_host(raw.strip().lstrip("*.").strip())
                if n and "." in n and not n.startswith("@"):
                    names.append(n)
        if len(names) > cap * 20:
            break
    return unique(names)[:cap]


def crtsh_lookup(
    domain: str, timeout: float = 8.0, *, client: httpx.Client | None = None
) -> tuple[list[str], str | None]:
    """(nazwy z Certificate Transparency dla ``%.domena``, błąd | None)."""
    d = registered_domain(domain) or clean_host(domain)
    if not d:
        return [], "crt.sh: nieprawidłowa domena"
    own = client is None
    c = client
    try:
        if c is None:
            c = httpx.Client(timeout=timeout, follow_redirects=True)
        resp = c.get(CRTSH_URL, params={"q": f"%.{d}", "output": "json"}, timeout=timeout)
        if resp.status_code >= 400:
            return [], f"crt.sh: HTTP {resp.status_code}"
        return parse_crtsh(resp.json()), None
    except httpx.TimeoutException:
        return [], "crt.sh: przekroczono czas"
    except ValueError:
        return [], "crt.sh: nieprawidłowa odpowiedź"
    except Exception as e:  # noqa: BLE001
        return [], f"crt.sh: błąd ({type(e).__name__})"
    finally:
        if own and c is not None:
            c.close()


def crtsh_domains(domain: str, timeout: float = 8.0, *, client: httpx.Client | None = None) -> list[str]:
    """Nazwy z certyfikatów (crt.sh, ``%.domena``): bez powtórzeń, bez ``*.``, najwyżej 200. Błąd → []."""
    return crtsh_lookup(domain, timeout, client=client)[0]
