"""Prosta pamięć podręczna JSON na dysku — wspólna dla modułów cech (fetch, dns, asn, rdap, tls, favicon, ocr, llm…).

- klucz (dowolny tekst) → sha1 → plik ``<katalog>/<sha1>.json``, więc nazwy plików są zawsze bezpieczne,
- wpis: ``{"k": klucz, "t": znacznik czasu zapisu, "v": wartość}``; wpis starszy niż ``ttl_h`` godzin = brak
  (``ttl_h <= 0`` — wpisy nie wygasają); ``get(..., allow_expired=True)`` zwraca także wpisy przeterminowane
  (tryb offline: lepsze stare dane niż żadne),
- zapis atomowy (plik tymczasowy w tym samym katalogu + ``os.replace``), bezpieczny dla wielu wątków;
  katalog tworzony leniwie przy pierwszym zapisie,
- uszkodzony lub obcy plik jest ignorowany, błąd zapisu tylko logowany — pamięć podręczna nigdy nie przerywa
  przetwarzania. Daty i inne obiekty spoza JSON zapisywane są jako tekst (``default=str``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class JsonCache:
    """Pamięć podręczna klucz → wartość JSON z czasem życia wpisów (TTL)."""

    def __init__(self, dir: Path | str, ttl_h: float = 72.0):  # noqa: A002 — nazwa parametru z kontraktu
        self.dir = Path(dir)
        self.ttl_h = float(ttl_h)
        self._lock = threading.Lock()
        self._dir_ready = False

    # ------------------------------------------------------------ ścieżki
    def path_for(self, key: str) -> Path:
        digest = hashlib.sha1(str(key).encode("utf-8", "surrogatepass")).hexdigest()
        return self.dir / f"{digest}.json"

    def _entry(self, key: str) -> dict[str, Any] | None:
        f = self.path_for(key)
        try:
            raw = f.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as e:
            log.debug("cache %s: nie można odczytać %s: %s", self.dir.name, f.name, e)
            return None
        try:
            entry = json.loads(raw)
        except ValueError:
            log.debug("cache %s: uszkodzony wpis %s — pomijam", self.dir.name, f.name)
            return None
        if not isinstance(entry, dict) or entry.get("k") != str(key) or "v" not in entry:
            return None
        return entry

    def _expired(self, entry: dict[str, Any]) -> bool:
        if self.ttl_h <= 0:
            return False
        t = entry.get("t")
        if not isinstance(t, (int, float)):
            return True
        return time.time() - float(t) > self.ttl_h * 3600.0

    # ------------------------------------------------------------ API
    def get(self, key: str, *, allow_expired: bool = False) -> Any | None:
        """Wartość dla klucza albo None (brak, wpis przeterminowany lub uszkodzony)."""
        entry = self._entry(key)
        if entry is None:
            return None
        if not allow_expired and self._expired(entry):
            return None
        return entry["v"]

    def age_h(self, key: str) -> float | None:
        """Wiek wpisu w godzinach (None, gdy brak wpisu)."""
        entry = self._entry(key)
        if entry is None or not isinstance(entry.get("t"), (int, float)):
            return None
        return max(0.0, (time.time() - float(entry["t"])) / 3600.0)

    def set(self, key: str, value: Any) -> None:
        """Zapisuje wartość (atomowo). Błędy zapisu są tylko logowane."""
        try:
            payload = json.dumps(
                {"k": str(key), "t": time.time(), "v": value}, ensure_ascii=False, default=str, separators=(",", ":")
            )
        except (TypeError, ValueError) as e:
            log.warning("cache %s: wartość nie daje się zapisać jako JSON: %s", self.dir.name, e)
            return
        target = self.path_for(key)
        with self._lock:
            tmp: str | None = None
            try:
                if not self._dir_ready:
                    self.dir.mkdir(parents=True, exist_ok=True)
                    self._dir_ready = True
                fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".tmp-", suffix=".json")
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(payload)
                os.replace(tmp, target)
                tmp = None
            except OSError as e:  # np. brak uprawnień, pełny dysk, plik chwilowo otwarty (Windows)
                log.warning("cache %s: nie można zapisać wpisu: %s", self.dir.name, e)
            finally:
                if tmp is not None:
                    Path(tmp).unlink(missing_ok=True)

    def delete(self, key: str) -> None:
        with self._lock:
            try:
                self.path_for(key).unlink(missing_ok=True)
            except OSError as e:
                log.debug("cache %s: nie można usunąć wpisu: %s", self.dir.name, e)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __repr__(self) -> str:
        return f"JsonCache({str(self.dir)!r}, ttl_h={self.ttl_h})"
