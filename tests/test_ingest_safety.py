"""Bezpieczeństwo i odporność wczytywania zbiorów: ścieżki z rekordów poza zbiorem, limity rozmiaru,
uszkodzone pliki (slad/ingest/loaders.py, pipeline.process_dataset). Bez sieci i LLM; dane tylko w tmp_path."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from PIL import Image

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

import slad.ingest.loaders as loaders  # noqa: E402
from slad import pipeline  # noqa: E402
from slad.ingest import load_dataset  # noqa: E402
from slad.settings import Settings  # noqa: E402
from slad.store import Workspace  # noqa: E402


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), "red").save(path)
    return path


def _spy_opened_paths(monkeypatch) -> list[Path]:
    """Zapisuje każdą ścieżkę otwieraną przez pathlib (read_bytes też przez open)."""
    opened: list[Path] = []
    real_open = Path.open

    def spy(self, *args, **kwargs):
        opened.append(self)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    return opened


def test_record_paths_outside_dataset_are_not_read(tmp_path, monkeypatch) -> None:
    root = tmp_path / "zbior"
    root.mkdir()
    secret_png = tmp_path / "sekret.png"  # poza katalogiem zbioru
    secret_png.write_bytes(b"\x89PNG-TAJNE")
    secret_html = tmp_path / "sekret.html"
    secret_html.write_text("<html>TAJNE</html>", encoding="utf-8")
    rows = [
        {"id": "a", "text": "reklama a", "image": "../sekret.png", "html": "../sekret.html"},
        {"id": "b", "text": "reklama b", "image": str(secret_png), "html": str(secret_html)},
        {"id": "c", "text": "reklama c", "image": "../../etc/hosts", "html": "../../etc/hosts"},
    ]
    (root / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    opened = _spy_opened_paths(monkeypatch)

    ms = {m.id: m for m in load_dataset(root)}

    assert set(ms) == {"a", "b", "c"}
    for m in ms.values():
        assert m.image_paths == [] and m.html_path is None, m.id
        assert m.meta.get("ingest_warnings"), m.id
    assert any("odrzucono" in w for w in ms["a"].meta["ingest_warnings"])
    assert any("odrzucono" in w for w in ms["b"].meta["ingest_warnings"])
    read = {p.resolve() for p in opened}
    assert secret_png.resolve() not in read and secret_html.resolve() not in read


def test_oversized_text_is_truncated_with_warning(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loaders, "MAX_TEXT_BYTES", 50)
    (tmp_path / "notatka.txt").write_text("A" * 500, encoding="utf-8")

    (m,) = load_dataset(tmp_path)

    assert m.text is not None and len(m.text) <= 50
    assert any("obcięto" in w for w in m.meta["ingest_warnings"])


def test_oversized_table_is_rejected_and_dataset_still_loads(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr(loaders, "MAX_TABLE_BYTES", 100)
    (tmp_path / "duzy.csv").write_text("id,text\n" + "".join(f"r{i},{'x' * 30}\n" for i in range(10)), encoding="utf-8")
    (tmp_path / "ok.jsonl").write_text(json.dumps({"id": "ok1", "text": "reklama"}) + "\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="slad.ingest.loaders"):
        ms = load_dataset(tmp_path)

    assert [m.id for m in ms] == ["ok1"]
    assert any("duzy.csv" in r.getMessage() and "za duży" in r.getMessage() for r in caplog.records)


def test_broken_json_in_folder_keeps_other_materials(tmp_path, caplog) -> None:
    _png(tmp_path / "ad1.png")
    _png(tmp_path / "ad2.png")
    (tmp_path / "uszkodzony.json").write_text("{[[[", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="slad.ingest.loaders"):
        ms = load_dataset(tmp_path)

    assert len(ms) == 2
    assert all(m.image_paths for m in ms)
    assert any("uszkodzony.json" in r.getMessage() for r in caplog.records)


def test_unreadable_file_is_reported_in_pipeline_summary(tmp_path) -> None:
    bad = tmp_path / "zepsuty.xlsx"
    bad.write_bytes(b"to nie jest skoroszyt")
    cfg = Settings()  # bez load_settings: nie czytamy .env
    cfg.llm.enabled = False
    cfg.network.enabled = False

    summary = pipeline.process_dataset(bad, Workspace(tmp_path / "ws"), cfg)

    assert summary["materials"] == 0
    assert len(summary["load_errors"]) == 1 and "zepsuty.xlsx" in summary["load_errors"][0]
    assert summary["error_samples"] == summary["load_errors"]
