"""Testy potoku (slad.pipeline): przetwarzanie zbioru, przyrostowość, odporność, stabilne id kampanii, analiza nowego.

Moduły ingest/ i link/ są podmieniane atrapami zgodnymi z kontraktem (docs/ARCHITEKTURA.md) — testy nie
zależą od ich implementacji. Testy integracyjne na prawdziwych modułach są pomijane, dopóki ich brakuje.
Bez sieci, LLM i twarzy.
"""

from __future__ import annotations

import sys
import threading
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

import slad.pipeline as pipeline  # noqa: E402
from slad.models import (  # noqa: E402
    Assignment,
    AssignmentStatus,
    Campaign,
    CampaignSummary,
    CorpusStats,
    Evidence,
    FeatureSet,
    Link,
    Material,
    Relation,
    SourceKind,
    TextFeatures,
)
from slad.pipeline import (  # noqa: E402
    analyze_new,
    attach_dates,
    load_run_info,
    process_dataset,
    rebuild_links,
    stabilize_campaign_ids,
)
from slad.settings import load_settings  # noqa: E402
from slad.store import Workspace  # noqa: E402

# ------------------------------------------------------------------ atrapy i dane


def _cfg(tmp_path: Path, workers: int = 2):
    cfg = load_settings(workspace=str(tmp_path / "ws"))
    cfg.network.enabled = False
    cfg.llm.enabled = False
    cfg.faces.enabled = False
    cfg.clip.enabled = False
    cfg.ocr.enabled = False
    cfg.workers = workers
    return cfg


def _materials() -> list[Material]:
    return [
        Material(
            id="m1",
            source=SourceKind.TEXT,
            text="Inwestuj z Bankiem X — 5000 zł dziennie",
            observed_at=datetime(2026, 11, 1),
            label="A",
        ),
        Material(
            id="m2",
            source=SourceKind.TEXT,
            text="Inwestuj z Bankiem X — 5000 zł dziennie!",
            observed_at=datetime(2026, 11, 5, tzinfo=UTC),
            label="A",
        ),
        Material(id="m3", source=SourceKind.URL, url="https://przyklad.pl/oferta", label="B"),
    ]


class FakeLink:
    """Atrapa slad.link: m1–m2 powiązane (jedna kampania), reszta osobno."""

    def __init__(self) -> None:
        self.build_calls: list[list[str]] = []
        self.summarize_calls: list[dict] = []
        self.assign_calls: list[str] = []
        self.fail_build = False
        self.fail_assign = False
        self.summarize_accepts_dates = True

    def build(self, features, cfg):
        ids = [fs.material_id for fs in features]
        self.build_calls.append(ids)
        if self.fail_build:
            raise RuntimeError("awaria łączenia")
        links, campaigns = [], []
        if "m1" in ids and "m2" in ids:
            ev = Evidence(
                kind="text_minhash", group="tekst", detail="niemal identyczny tekst", similarity=0.9, weight=5.0
            )
            links.append(Link(a="m1", b="m2", score=9.0, relation=Relation.VARIANT, evidence=[ev], coverage=1.0))
            campaigns.append(Campaign(id="K001", name="Inwestycje — Bank X", members=["m1", "m2"], cohesion=9.0))
        if "new-1" in ids and "m3" in ids:
            ev = Evidence(kind="domain", group="infrastruktura", detail="wspólna domena przyklad.pl", weight=5.0)
            links.append(Link(a="m3", b="new-1", score=5.0, relation=Relation.RELATED, evidence=[ev]))
            campaigns.append(Campaign(id=f"K{len(campaigns) + 1:03d}", name="Przykład", members=["m3", "new-1"]))
        return links, campaigns, CorpusStats(n=len(ids))

    def summarize(self, cid, members, features_by_id, links, **kwargs):
        if not self.summarize_accepts_dates and "dates" in kwargs:
            raise TypeError("summarize() got an unexpected keyword argument 'dates'")
        self.summarize_calls.append({"cid": cid, "members": list(members), "links": len(links), **kwargs})
        dates = [kwargs.get("dates", {}).get(m) for m in members]
        dates = [d for d in dates if d is not None]
        summary = CampaignSummary(
            first_seen=min(dates, key=pipeline._utc_key) if dates else None,
            last_seen=max(dates, key=pipeline._utc_key) if dates else None,
        )
        return Campaign(id=cid, name="Inwestycje — Bank X", members=list(members), summary=summary)

    def assign(self, fs, ws, cfg):
        self.assign_calls.append(fs.material_id)
        if self.fail_assign:
            raise RuntimeError("awaria przypisania")
        known = ws.campaigns()
        if known and "Bank" in fs.text.text:
            return Assignment(
                material_id=fs.material_id,
                campaign_id=known[0].id,
                score=8.0,
                status=AssignmentStatus.ASSIGNED,
                explanation="Ten sam tekst co kampania K001.",
            )
        return Assignment(material_id=fs.material_id, explanation="Brak podobnych materiałów.")


@pytest.fixture
def fake_link(monkeypatch) -> FakeLink:
    fake = FakeLink()
    link_mod = types.ModuleType("slad.link")
    link_mod.__path__ = []  # pakiet: inne podmoduły → czytelny ImportError
    link_mod.build = fake.build
    summarize_mod = types.ModuleType("slad.link.summarize")
    summarize_mod.summarize = fake.summarize
    assign_mod = types.ModuleType("slad.link.assign")
    assign_mod.assign = fake.assign
    monkeypatch.setitem(sys.modules, "slad.link", link_mod)
    monkeypatch.setitem(sys.modules, "slad.link.summarize", summarize_mod)
    monkeypatch.setitem(sys.modules, "slad.link.assign", assign_mod)
    return fake


@pytest.fixture
def dataset(monkeypatch) -> list[Material]:
    """Atrapa slad.ingest.load_dataset — zwraca (modyfikowalną) listę materiałów."""
    items = _materials()
    ingest_mod = types.ModuleType("slad.ingest")
    ingest_mod.__path__ = []
    ingest_mod.load_dataset = lambda path: [m.model_copy(deep=True) for m in items]
    monkeypatch.setitem(sys.modules, "slad.ingest", ingest_mod)
    return items


@pytest.fixture
def extract_calls(monkeypatch) -> list[str]:
    """Atrapa slad.features.extract (szybka, deterministyczna); m-bad rzuca wyjątkiem."""
    calls: list[str] = []
    lock = threading.Lock()

    def fake_extract(material, ctx):
        with lock:
            calls.append(material.id)
        if material.id == "m-bad":
            raise ValueError("uszkodzony plik")
        text = material.text or material.url or ""
        return FeatureSet(material_id=material.id, text=TextFeatures(text=text), available=["tekst"] if text else [])

    monkeypatch.setattr(pipeline, "extract", fake_extract)
    return calls


# ------------------------------------------------------------------ process_dataset


def test_process_dataset_writes_workspace(tmp_path, fake_link, dataset, extract_calls):
    cfg = _cfg(tmp_path)
    ws = Workspace(tmp_path / "ws")
    progress: list[tuple[int, int, str]] = []

    summary = process_dataset(tmp_path / "zbior.csv", ws, cfg, progress=lambda *a: progress.append(a))

    assert summary["materials"] == 3
    assert summary["extracted"] == 3 and summary["cached"] == 0
    assert summary["links"] == 1 and summary["relations"] == {"wariant": 1}
    assert summary["campaigns"] == 1 and summary["clustered"] == 2
    assert summary["errors"] == 0 and "link_error" not in summary
    for name in ("materials.jsonl", "links.jsonl", "campaigns.json", "stats.json", "run.json"):
        assert ws.path(name).exists(), name
    assert len(list((ws.root / "features").glob("*.json"))) == 3
    assert progress[0] == (0, 3, "") and progress[-1][:2] == (3, 3)
    assert sorted(fake_link.build_calls[0]) == ["m1", "m2", "m3"]
    # daty kampanii: summarize(..., dates=...) — mieszane strefy czasowe nie przeszkadzają
    assert "dates" in fake_link.summarize_calls[0]
    c = ws.campaigns()[0]
    assert c.summary.first_seen.date().isoformat() == "2026-11-01"
    assert c.summary.last_seen.date().isoformat() == "2026-11-05"
    info = load_run_info(ws)
    assert info["materials"] == 3 and info["settings"]["network"] is False and "finished_at" in info


def test_process_dataset_is_incremental(tmp_path, fake_link, dataset, extract_calls):
    cfg = _cfg(tmp_path)
    ws = Workspace(tmp_path / "ws")
    process_dataset(tmp_path / "zbior", ws, cfg)
    assert sorted(extract_calls) == ["m1", "m2", "m3"]

    extract_calls.clear()
    again = process_dataset(tmp_path / "zbior", Workspace(tmp_path / "ws"), cfg)
    assert extract_calls == [] and again["cached"] == 3 and again["extracted"] == 0
    assert again["links"] == 1  # powiązania liczone z cech z zapisu

    dataset[2] = dataset[2].model_copy(update={"text": "zmieniona treść"})  # zmieniony materiał → od nowa
    dataset[0] = dataset[0].model_copy(update={"label": "Z"})  # sama etykieta nie jest cechą
    process_dataset(tmp_path / "zbior", Workspace(tmp_path / "ws"), cfg)
    assert extract_calls == ["m3"]

    extract_calls.clear()
    forced = process_dataset(tmp_path / "zbior", Workspace(tmp_path / "ws"), cfg, force=True)
    assert sorted(extract_calls) == ["m1", "m2", "m3"] and forced["extracted"] == 3


def test_process_dataset_limit_and_duplicate_ids(tmp_path, fake_link, dataset, extract_calls):
    dataset.append(dataset[0].model_copy())
    summary = process_dataset(tmp_path / "zbior", Workspace(tmp_path / "ws"), _cfg(tmp_path), limit=2)
    assert summary["duplicate_ids"] == 1
    assert summary["materials"] == 2
    assert [m.id for m in Workspace(tmp_path / "ws").materials()] == ["m1", "m2"]


def test_bad_material_does_not_stop_processing(tmp_path, fake_link, dataset, extract_calls):
    dataset.append(Material(id="m-bad", source=SourceKind.IMAGE, image_paths=["/nie/ma/obrazu.png"]))
    ws = Workspace(tmp_path / "ws")
    summary = process_dataset(tmp_path / "zbior", ws, _cfg(tmp_path, workers=3))
    assert summary["materials"] == 4 and summary["features"] == 4
    assert summary["errors"] == 1 and summary["error_samples"][0].startswith("m-bad: ekstrakcja cech")
    assert ws.features("m-bad").errors  # zapisany z błędem, nie pominięty
    assert summary["campaigns"] == 1


def test_link_failure_keeps_features(tmp_path, fake_link, dataset, extract_calls):
    fake_link.fail_build = True
    ws = Workspace(tmp_path / "ws")
    summary = process_dataset(tmp_path / "zbior", ws, _cfg(tmp_path))
    assert "awaria łączenia" in summary["link_error"]
    assert len(ws.all_features()) == 3
    assert load_run_info(ws)["link_error"]


def test_empty_dataset_leaves_workspace_untouched(tmp_path, fake_link, dataset, extract_calls):
    ws = Workspace(tmp_path / "ws")
    process_dataset(tmp_path / "zbior", ws, _cfg(tmp_path))
    dataset.clear()
    summary = process_dataset(tmp_path / "pusty", Workspace(tmp_path / "ws"), _cfg(tmp_path))
    assert summary["materials"] == 0 and summary["campaigns"] == 0
    assert len(Workspace(tmp_path / "ws").materials()) == 3


def test_progress_is_called_in_calling_thread(tmp_path, fake_link, dataset, extract_calls):
    dataset.extend(Material(id=f"x{i}", source=SourceKind.TEXT, text=f"tekst {i}") for i in range(8))
    threads: set[str] = set()
    done: list[int] = []

    def progress(d, total, mid):
        threads.add(threading.current_thread().name)
        done.append(d)

    process_dataset(tmp_path / "zbior", Workspace(tmp_path / "ws"), _cfg(tmp_path, workers=4), progress=progress)
    assert threads == {threading.current_thread().name}
    assert done == list(range(0, 12))


def test_rebuild_links_invalidates_text_index(tmp_path, fake_link, dataset, extract_calls):
    ws = Workspace(tmp_path / "ws")
    process_dataset(tmp_path / "zbior", ws, _cfg(tmp_path))
    ws.path("text_index.pkl").write_bytes(b"stary indeks")
    summary = rebuild_links(ws, _cfg(tmp_path))
    assert summary["links"] == 1 and summary["campaigns"] == 1
    assert not ws.path("text_index.pkl").exists()


# ------------------------------------------------------------------ id kampanii i daty


def _camp(cid: str, members: list[str]) -> Campaign:
    return Campaign(id=cid, name=f"kampania {cid}", members=members)


def test_stabilize_keeps_ids_after_renumbering():
    old = [_camp("K001", ["a", "b", "c"]), _camp("K002", ["d", "e"])]
    new = [_camp("K001", ["d", "e", "x"]), _camp("K002", ["a", "b", "c"]), _camp("K003", ["y", "z"])]
    out = stabilize_campaign_ids(new, old, current_ids="a b c d e x y z".split())
    assert [(c.id, c.members) for c in out] == [
        ("K002", ["d", "e", "x"]),
        ("K001", ["a", "b", "c"]),
        ("K003", ["y", "z"]),  # nowa kampania: numer powyżej dotychczasowych
    ]
    assert len({c.id for c in out}) == len(out)


def test_stabilize_merge_and_split_keep_unique_ids():
    old = [_camp("K001", ["a", "b", "c", "d"]), _camp("K002", ["e", "f"])]
    new = [_camp("K001", ["a", "b", "e", "f"]), _camp("K002", ["c", "d"])]
    out = stabilize_campaign_ids(new, old)
    assert len({c.id for c in out}) == 2
    assert {c.id for c in out} <= {"K001", "K002", "K003", "K004"}


def test_stabilize_leaves_unrelated_or_foreign_ids_alone():
    old = [_camp("K001", ["a", "b"])]
    new = [_camp("K001", ["p", "q"]), _camp("K002", ["r", "s"])]
    assert stabilize_campaign_ids(new, old) == new  # brak pokrycia → numeracja od nowa
    assert stabilize_campaign_ids(new, old, current_ids=["p", "q", "r", "s"]) == new  # inny zbiór
    odd = [_camp("C-7", ["a", "b"])]
    assert stabilize_campaign_ids(odd, old) == odd  # nieznany format id
    assert stabilize_campaign_ids([], old) == [] and stabilize_campaign_ids(new, []) == new


def test_stabilize_renames_fallback_names():
    old = [_camp("K001", ["a", "b"]), _camp("K002", ["c", "d"])]
    new = [
        Campaign(id="K001", name="Kampania K001", members=["c", "d"]),
        Campaign(id="K002", name="Jan Kowalski · inwestycje", members=["a", "b"]),
    ]
    out = stabilize_campaign_ids(new, old)
    assert [(c.id, c.name) for c in out] == [("K002", "Kampania K002"), ("K001", "Jan Kowalski · inwestycje")]


def test_build_extensions_dates_and_progress(tmp_path, fake_link, dataset, extract_calls):
    """build(features, cfg, dates=, progress=) — przekazywane, gdy build je przyjmuje; bez drugiego summarize."""
    seen: dict = {}

    def build(features, cfg, dates=None, progress=None):
        seen["dates"] = dates
        if progress is not None:
            progress("powiązania", 1, 1)
        return fake_link.build(features, cfg)

    sys.modules["slad.link"].build = build  # moduł-atrapa z fixture (znika po teście)
    stages: list[tuple] = []
    ws = Workspace(tmp_path / "ws")
    process_dataset(tmp_path / "zbior", ws, _cfg(tmp_path), stage_progress=lambda *a: stages.append(a))

    assert set(seen["dates"]) == {"m1", "m2"}
    assert stages == [("powiązania", 1, 1)]
    assert fake_link.summarize_calls == []
    assert ws.campaigns()[0].summary.first_seen is not None  # daty uzupełnione nawet bez summarize


def test_attach_dates_without_summarize_dates_support(fake_link):
    fake_link.summarize_accepts_dates = False
    materials = _materials()
    campaigns = [_camp("K001", ["m1", "m2"]), _camp("K002", ["m3"])]
    out = attach_dates(campaigns, materials, [], [])
    assert out[0].summary.first_seen == datetime(2026, 11, 1)
    assert out[0].summary.last_seen == datetime(2026, 11, 5, tzinfo=UTC)
    assert out[1].summary.first_seen is None  # brak dat → bez zmian


# ------------------------------------------------------------------ analiza nowego materiału


def test_analyze_new_returns_assignment_without_changing_base(tmp_path, fake_link, dataset, extract_calls):
    cfg = _cfg(tmp_path)
    ws = Workspace(tmp_path / "ws")
    process_dataset(tmp_path / "zbior", ws, cfg)
    new = Material(id="new-1", source=SourceKind.TEXT, text="Bank X: 5000 zł dziennie")

    fs, assignment = analyze_new(new, ws, cfg)

    assert isinstance(fs, FeatureSet) and fs.material_id == "new-1"
    assert isinstance(assignment, Assignment)
    assert assignment.status == AssignmentStatus.ASSIGNED and assignment.campaign_id == "K001"
    assert ws.material("new-1") is None and ws.features("new-1") is None


def test_analyze_new_add_persists_and_rebuilds(tmp_path, fake_link, dataset, extract_calls):
    cfg = _cfg(tmp_path)
    ws = Workspace(tmp_path / "ws")
    process_dataset(tmp_path / "zbior", ws, cfg)
    new = Material(id="new-1", source=SourceKind.URL, url="https://przyklad.pl/inna")

    fs, assignment = analyze_new(new, ws, cfg, add=True)

    assert assignment.status == AssignmentStatus.UNLINKED  # stan bazy PRZED dodaniem
    assert ws.material("new-1") is not None and ws.features("new-1") is not None
    assert sorted(fake_link.build_calls[-1]) == ["m1", "m2", "m3", "new-1"]
    assert {c.id for c in ws.campaigns()} == {"K001", "K002"}
    assert assignment.campaign_id == "K002" and "K002" in assignment.explanation
    info = load_run_info(ws)  # przegląd w UI nie pokazuje nieaktualnych liczb
    assert info["materials"] == 4 and info["campaigns"] == 2 and "relinked_at" in info


def test_analyze_new_survives_assign_failure(tmp_path, fake_link, extract_calls):
    fake_link.fail_assign = True
    cfg = _cfg(tmp_path)
    new = Material(id="new-2", source=SourceKind.TEXT, text="cokolwiek")
    fs, assignment = analyze_new(new, Workspace(tmp_path / "ws"), cfg)
    assert assignment.status == AssignmentStatus.UNLINKED
    assert "Nie udało się porównać" in assignment.explanation


# ------------------------------------------------------------------ integracja z prawdziwymi modułami


def _real_modules():
    ingest = pytest.importorskip("slad.ingest")
    link = pytest.importorskip("slad.link")
    if not hasattr(link, "build"):
        pytest.skip("slad.link.build jeszcze nie istnieje")
    return ingest


def _tiny_images(folder: Path) -> Path:
    folder.mkdir(parents=True)
    for name, color, shift in (("reklama_a.png", (200, 30, 30), 0), ("reklama_b.png", (200, 30, 30), 2)):
        im = Image.new("RGB", (96, 96), color)
        ImageDraw.Draw(im).rectangle((10 + shift, 10, 60 + shift, 50), fill=(250, 250, 250))
        im.save(folder / name)
    im = Image.new("RGB", (96, 96), (20, 60, 200))
    ImageDraw.Draw(im).ellipse((20, 20, 80, 80), fill=(0, 0, 0))
    im.save(folder / "inna.png")
    return folder


def test_process_dataset_with_real_modules(tmp_path):
    ingest = _real_modules()
    data = _tiny_images(tmp_path / "zbior")
    cfg = _cfg(tmp_path)
    ws = Workspace(tmp_path / "ws")

    summary = process_dataset(data, ws, cfg)

    assert summary["materials"] == 3 and summary["features"] == 3
    assert "link_error" not in summary, summary.get("link_error")
    for name in ("materials.jsonl", "links.jsonl", "campaigns.json", "stats.json"):
        assert ws.path(name).exists(), name
    assert len(list((ws.root / "features").glob("*.json"))) == 3
    assert all("obraz" in fs.available for fs in ws.all_features())

    pytest.importorskip("slad.link.assign")
    new = ingest.material_from_input(image_path=str(data / "reklama_a.png"))
    fs, assignment = analyze_new(new, ws, cfg)
    assert isinstance(fs, FeatureSet) and isinstance(assignment, Assignment)
