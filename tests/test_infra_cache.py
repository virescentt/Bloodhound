"""JsonCache: zapis/odczyt, TTL, odporność na uszkodzone wpisy, wielowątkowość."""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.features.infra import cache as cache_mod  # noqa: E402
from slad.features.infra.cache import JsonCache  # noqa: E402


def test_roundtrip_hashed_names_and_lazy_dir(tmp_path):
    d = tmp_path / "fetch"
    c = JsonCache(d, ttl_h=1)
    assert not d.exists()  # katalog tworzony dopiero przy zapisie
    assert c.get("https://example.com/a?b=1") is None
    c.set(
        "https://example.com/a?b=1", {"x": 1, "when": datetime(2025, 1, 2, tzinfo=UTC), "items": [1, 2], "pl": "żółć"}
    )
    assert c.get("https://example.com/a?b=1") == {
        "x": 1,
        "when": "2025-01-02 00:00:00+00:00",  # datetime → tekst (default=str)
        "items": [1, 2],
        "pl": "żółć",
    }
    files = list(d.glob("*.json"))
    assert len(files) == 1 and len(files[0].stem) == 40  # sha1 klucza
    assert "https://example.com/a?b=1" in c
    assert "inny" not in c


def test_ttl_expiry_and_allow_expired(tmp_path, monkeypatch):
    c = JsonCache(tmp_path, ttl_h=1)
    c.set("k", {"v": 1})
    now = time.time()
    monkeypatch.setattr(cache_mod.time, "time", lambda: now + 2 * 3600)
    assert c.get("k") is None
    assert c.get("k", allow_expired=True) == {"v": 1}
    assert c.age_h("k") == pytest.approx(2.0, abs=0.05)


def test_ttl_zero_never_expires(tmp_path, monkeypatch):
    c = JsonCache(tmp_path, ttl_h=0)
    c.set("k", 5)
    now = time.time()
    monkeypatch.setattr(cache_mod.time, "time", lambda: now + 10_000 * 3600)
    assert c.get("k") == 5


def test_overwrite_and_delete(tmp_path):
    c = JsonCache(tmp_path, ttl_h=1)
    c.set("k", 1)
    c.set("k", 2)
    assert c.get("k") == 2
    c.delete("k")
    assert c.get("k") is None
    c.delete("k")  # brak wpisu — bez wyjątku


def test_corrupt_or_foreign_entries_are_ignored(tmp_path):
    c = JsonCache(tmp_path, ttl_h=1)
    c.set("k", 1)
    c.path_for("k").write_text("{zepsuty json", encoding="utf-8")
    assert c.get("k") is None
    c.path_for("k").write_text(json.dumps({"k": "inny-klucz", "t": time.time(), "v": 5}), encoding="utf-8")
    assert c.get("k") is None
    c.path_for("k").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert c.get("k") is None


def test_unserializable_value_and_unwritable_dir_do_not_raise(tmp_path):
    c = JsonCache(tmp_path, ttl_h=1)
    c.set("k", {(1, 2): "klucz-krotka"})  # TypeError w json.dumps — tylko log
    assert c.get("k") is None
    blocker = tmp_path / "plik"
    blocker.write_text("x")
    bad = JsonCache(blocker / "podkatalog", ttl_h=1)  # rodzic jest plikiem → mkdir się nie uda
    bad.set("k", 1)
    assert bad.get("k") is None


def test_thread_safe_parallel_writes(tmp_path):
    c = JsonCache(tmp_path / "c", ttl_h=1)
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(40):
                c.set("wspolny", {"n": n, "i": i})
                c.set(f"k-{n}-{i}", i)
                assert c.get(f"k-{n}-{i}") == i
                v = c.get("wspolny")
                assert v is None or set(v) == {"n", "i"}
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert c.get("k-7-39") == 39
    assert not list((tmp_path / "c").glob(".tmp-*"))  # żadnych pozostałości po zapisie atomowym
