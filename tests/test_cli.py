"""Testy CLI (slad.cli) przez typer.testing.CliRunner — moduły przetwarzania podmienione atrapami, bez sieci."""

from __future__ import annotations

import json
import logging
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

import slad.cli as cli  # noqa: E402
import slad.pipeline as pipeline  # noqa: E402
from slad.cli import app, detect_input, streamlit_command  # noqa: E402
from slad.models import (  # noqa: E402
    Assignment,
    AssignmentStatus,
    Campaign,
    CampaignSummary,
    Evidence,
    FeatureSet,
    Link,
    Material,
    Relation,
    SourceKind,
    TextFeatures,
)
from slad.settings import ROOT  # noqa: E402
from slad.store import Workspace  # noqa: E402

runner = CliRunner(env={"COLUMNS": "200"})  # szeroka konsola rich — tabele bez zawijania w asercjach

COMMANDS = ("run", "ingest", "link", "analyze", "campaigns", "export", "eval", "synthetic", "models", "ui", "demo")


@pytest.fixture(autouse=True)
def _restore_logging():
    """CLI konfiguruje logowanie (RichHandler, poziom) — przywracamy stan, by nie wpływać na inne testy."""
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def fake_module(monkeypatch, name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def make_workspace(root: Path) -> Workspace:
    ws = Workspace(root)
    ws.save_materials(
        [
            Material(id="m1", source=SourceKind.URL, url="https://zysk-teraz.pl/a", platform="facebook"),
            Material(id="m2", source=SourceKind.TEXT, text="Inwestuj [teraz] 5000 zł dziennie"),
        ]
    )
    for mid in ("m1", "m2"):
        ws.save_features(FeatureSet(material_id=mid, text=TextFeatures(text="inwestuj"), available=["tekst"]))
    ev = Evidence(
        kind="text_minhash", group="tekst", detail="niemal identyczny tekst [98%]", similarity=0.98, weight=5.0
    )
    ws.save_links([Link(a="m1", b="m2", score=9.0, relation=Relation.VARIANT, evidence=[ev])])
    ws.save_campaigns(
        [
            Campaign(
                id="K001",
                name="Inwestycje — Bank X",
                members=["m1", "m2"],
                cohesion=9.0,
                summary=CampaignSummary(persons=[("Jan Kowalski", 2)], first_seen=datetime(2026, 11, 1)),
                key_evidence=[ev],
            )
        ]
    )
    return ws


# ------------------------------------------------------------------ pomoc i wykrywanie wejścia


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for command in COMMANDS:
        assert command in result.output
    for command in COMMANDS:
        sub = runner.invoke(app, [command, "--help"])
        assert sub.exit_code == 0, (command, sub.output)


def test_detect_input(tmp_path):
    img = tmp_path / "zrzut.png"
    Image.new("RGB", (8, 8), "red").save(img)
    noext = tmp_path / "obraz_bez_rozszerzenia"
    noext.write_bytes(img.read_bytes())
    html = tmp_path / "strona.htm"
    html.write_text("<html><body>oferta</body></html>", encoding="utf-8")
    sniffed = tmp_path / "zapis"
    sniffed.write_text("<!DOCTYPE html><html><title>x</title></html>", encoding="utf-8")
    txt = tmp_path / "reklama.txt"
    txt.write_text("Zarabiaj 5000 zł dziennie", encoding="utf-8")

    assert detect_input("https://zysk.pl/a?b=1") == {"url": "https://zysk.pl/a?b=1"}
    assert detect_input("zysk-teraz.pl/oferta") == {"url": "http://zysk-teraz.pl/oferta"}
    assert detect_input(str(img)) == {"image_path": str(img.resolve())}
    assert detect_input(str(noext)) == {"image_path": str(noext.resolve())}
    assert detect_input(str(html)) == {"html_path": str(html.resolve())}
    assert detect_input(str(sniffed)) == {"html_path": str(sniffed.resolve())}
    assert detect_input(str(txt)) == {"text": "Zarabiaj 5000 zł dziennie"}
    assert detect_input("  Zarabiaj z Bankiem X 5000 zł dziennie! ") == {
        "text": "Zarabiaj z Bankiem X 5000 zł dziennie!"
    }
    assert detect_input("zysk.pl", kind="text") == {"text": "zysk.pl"}
    assert detect_input("zysk.pl", kind="URL") == {"url": "http://zysk.pl"}
    assert detect_input(str(txt), kind="text") == {"text": "Zarabiaj 5000 zł dziennie"}
    assert detect_input("x" * 5000)["text"].startswith("xxx")  # długi tekst nie jest traktowany jak ścieżka
    with pytest.raises(ValueError, match="katalog"):
        detect_input(str(tmp_path))
    with pytest.raises(ValueError, match="Nie znaleziono"):
        detect_input(str(tmp_path / "brak.png"), kind="image")
    with pytest.raises(ValueError, match="Nieznany typ"):
        detect_input("abc", kind="wideo")
    with pytest.raises(ValueError, match="Puste"):
        detect_input("   ")


def test_streamlit_command():
    cmd = streamlit_command(Path("/app/slad/ui/app.py"), 8502, "0.0.0.0")
    assert cmd[:5] == [sys.executable, "-m", "streamlit", "run", "/app/slad/ui/app.py"]
    joined = " ".join(cmd)
    assert "--server.port 8502" in joined and "--server.headless true" in joined
    assert "--browser.gatherUsageStats false" in joined and "--server.address 0.0.0.0" in joined
    assert "--server.address" not in " ".join(streamlit_command(Path("app.py")))


# ------------------------------------------------------------------ run / link / analyze


def test_run_applies_overrides_and_prints_summary(tmp_path, monkeypatch):
    data = tmp_path / "zbior.csv"
    data.write_text("id,text\n1,abc\n", encoding="utf-8")
    seen: dict = {}

    def fake_process(path, ws, cfg, progress=None, limit=None, force=False, stage_progress=None):
        seen.update(path=path, root=ws.root, cfg=cfg, limit=limit, force=force)
        progress(0, 2, "")
        progress(1, 2, "rekord-[1]")  # id z nawiasami nie może psuć znaczników rich
        progress(2, 2, "rekord-2")
        stage_progress("powiązania", 3, 10)
        stage_progress("powiązania", 10, 10)
        return {"materials": 2, "features": 2, "extracted": 2, "cached": 0, "links": 1, "relations": {"wariant": 1},
                "campaigns": 1, "clustered": 2, "errors": 0, "time_s": 0.5}  # fmt: skip

    monkeypatch.setattr(pipeline, "process_dataset", fake_process)
    ws_dir = tmp_path / "ws"
    args = ["run", str(data), "-w", str(ws_dir), "--offline", "--no-llm", "--no-faces", "--limit", "5", "-j", "2"]
    result = runner.invoke(app, [*args, "--force"])

    assert result.exit_code == 0, result.output
    cfg = seen["cfg"]
    assert seen["root"] == ws_dir.resolve() and seen["limit"] == 5 and seen["force"] is True
    assert cfg.network.enabled is False and cfg.llm.enabled is False and cfg.faces.enabled is False
    assert cfg.workers == 2
    assert cfg.network.timeout_s > 0 and cfg.link.weights  # reszta sekcji z settings.yaml zachowana
    assert "Podsumowanie" in result.output and "wariant 1" in result.output


def test_run_missing_dataset_and_failure(tmp_path, monkeypatch):
    result = runner.invoke(app, ["run", str(tmp_path / "nie-ma"), "-w", str(tmp_path / "ws")])
    assert result.exit_code == 2 and "Nie znaleziono zbioru" in result.output

    def boom(*args, **kwargs):
        raise ValueError("nieznany format pliku")

    monkeypatch.setattr(pipeline, "process_dataset", boom)
    (tmp_path / "zbior.xyz").write_text("?", encoding="utf-8")
    result = runner.invoke(app, ["run", str(tmp_path / "zbior.xyz"), "-w", str(tmp_path / "ws")])
    assert result.exit_code == 1
    assert "Błąd (przetwarzanie)" in result.output and "nieznany format" in result.output


def test_run_reports_link_error(tmp_path, monkeypatch):
    (tmp_path / "zbior").mkdir()
    monkeypatch.setattr(
        pipeline, "process_dataset", lambda *a, **k: {"materials": 1, "link_error": "RuntimeError: awaria"}
    )
    result = runner.invoke(app, ["run", str(tmp_path / "zbior"), "-w", str(tmp_path / "ws")])
    assert result.exit_code == 1 and "awaria" in result.output


def test_link_command(tmp_path, monkeypatch):
    ws_dir = tmp_path / "ws"
    assert runner.invoke(app, ["link", "-w", str(ws_dir)]).exit_code == 1  # pusty workspace
    make_workspace(ws_dir)
    calls = []

    def fake_rebuild(ws, cfg, stage_progress=None):
        calls.append(ws.root)
        stage_progress("powiązania", 5, 5)
        return {"links": 3, "campaigns": 1}

    monkeypatch.setattr(pipeline, "rebuild_links", fake_rebuild)
    result = runner.invoke(app, ["link", "-w", str(ws_dir)])
    assert result.exit_code == 0, result.output
    assert calls == [ws_dir.resolve()] and "Powiązania" in result.output


def test_analyze_prints_assignment_and_evidence(tmp_path, monkeypatch):
    ws_dir = tmp_path / "ws"
    make_workspace(ws_dir)
    made: dict = {}

    def material_from_input(**kwargs):
        made.update(kwargs)
        return Material(id="new-abc", source=SourceKind.TEXT, text=kwargs.get("text"))

    fake_module(monkeypatch, "slad.ingest", material_from_input=material_from_input)
    ev = Evidence(kind="tracker_id", group="infrastruktura", detail="wspólny Facebook Pixel 123 [fb]", weight=6.0)

    def fake_analyze(material, ws, cfg, add=False):
        made["add"] = add
        made["offline"] = not cfg.network.enabled
        nearest = [Link(a="m1", b="new-abc", score=11.0, relation=Relation.VARIANT, evidence=[ev])]
        return FeatureSet(material_id=material.id, available=["tekst"]), Assignment(
            material_id=material.id,
            campaign_id="K001",
            score=11.0,
            status=AssignmentStatus.ASSIGNED,
            nearest=nearest,
            explanation="Wspólny piksel śledzący z kampanią K001.",
        )

    monkeypatch.setattr(pipeline, "analyze_new", fake_analyze)
    result = runner.invoke(app, ["analyze", "Inwestuj 5000 zł", "-w", str(ws_dir), "--offline", "--add"])

    assert result.exit_code == 0, result.output
    assert made["text"] == "Inwestuj 5000 zł" and made["add"] is True and made["offline"] is True
    out = result.output
    assert "Przypisano do kampanii K001" in out and "Inwestycje" in out
    assert "Facebook Pixel 123 [fb]" in out  # treść przesłanki bez interpretacji jako znaczniki rich
    assert "m1" in out and "Dodano do bazy" in out


def test_analyze_rejects_bad_input(tmp_path):
    result = runner.invoke(app, ["analyze", str(tmp_path / "brak.png"), "--as", "image", "-w", str(tmp_path / "ws")])
    assert result.exit_code == 2 and "Nie znaleziono pliku" in result.output


# ------------------------------------------------------------------ wyniki: campaigns / export / eval


def test_campaigns_list_and_details(tmp_path):
    ws_dir = tmp_path / "ws"
    empty = runner.invoke(app, ["campaigns", "-w", str(ws_dir)])
    assert empty.exit_code == 0 and "Brak kampanii" in empty.output
    make_workspace(ws_dir)

    listing = runner.invoke(app, ["campaigns", "-w", str(ws_dir)])
    assert listing.exit_code == 0, listing.output
    assert "K001" in listing.output and "Inwestycje" in listing.output and "Jan Kowalski" in listing.output

    details = runner.invoke(app, ["campaigns", "k001", "-w", str(ws_dir)])
    assert details.exit_code == 0, details.output
    assert "niemal identyczny tekst [98%]" in details.output and "zysk-teraz.pl" in details.output

    missing = runner.invoke(app, ["campaigns", "K999", "-w", str(ws_dir)])
    assert missing.exit_code == 2 and "Nie ma kampanii" in missing.output


def test_export_all_formats(tmp_path, monkeypatch):
    fake_module(monkeypatch, "slad.link.indicators", indicators=lambda fs: {"keyword": {"inwestuj"}})
    ws_dir = tmp_path / "ws"
    make_workspace(ws_dir)
    out = tmp_path / "eksport"
    result = runner.invoke(app, ["export", "-w", str(ws_dir), "--format", "all", "--out", str(out)])
    assert result.exit_code == 0, result.output
    for name in ("nodes_materials.csv", "rels_links.csv", "import.cypher", "bloodhound.graphml", "bloodhound.json"):
        assert (out / name).exists(), name

    single = runner.invoke(app, ["export", "-w", str(ws_dir), "-f", "json", "-o", str(tmp_path / "x" / "graf.json")])
    assert single.exit_code == 0 and (tmp_path / "x" / "graf.json").exists()
    default = runner.invoke(app, ["export", "-w", str(ws_dir), "-f", "graphml"])
    assert default.exit_code == 0 and (ws_dir / "export" / "bloodhound.graphml").exists()
    bad = runner.invoke(app, ["export", "-w", str(ws_dir), "-f", "pdf"])
    assert bad.exit_code == 2 and "Nieznany format" in bad.output


def test_eval_saves_report(tmp_path, monkeypatch):
    ws_dir = tmp_path / "ws"
    make_workspace(ws_dir)
    fake_module(monkeypatch, "slad.eval")
    fake_module(
        monkeypatch,
        "slad.eval.run",
        evaluate=lambda ws, cfg: {"pairwise": {"f1": 0.9}, "n": len(ws.materials())},
        format_report=lambda report: f"F1 par: {report['pairwise']['f1']:.2f} [n={report['n']}]",
    )
    result = runner.invoke(app, ["eval", "-w", str(ws_dir)])
    assert result.exit_code == 0, result.output
    assert "F1 par: 0.90 [n=2]" in result.output
    assert json.loads((ws_dir / "report.json").read_text(encoding="utf-8")) == {"pairwise": {"f1": 0.9}, "n": 2}


# ------------------------------------------------------------------ synthetic / models / ui / demo


def test_synthetic_calls_generator(tmp_path, monkeypatch):
    calls = []

    def generate(out_dir, seed=7, n_campaigns=8, noise=12):
        calls.append((Path(out_dir), seed, n_campaigns, noise))
        return Path(out_dir) / "dataset.jsonl"

    fake_module(monkeypatch, "slad.ingest.synthetic", generate=generate)
    result = runner.invoke(app, ["synthetic", "--out", str(tmp_path / "syn"), "--seed", "3", "--noise", "4"])
    assert result.exit_code == 0, result.output
    assert calls == [((tmp_path / "syn").resolve(), 3, 8, 4)]
    assert "dataset.jsonl" in result.output


def test_synthetic_real_generator(tmp_path):
    pytest.importorskip("slad.ingest.synthetic")
    result = runner.invoke(app, ["synthetic", "--out", str(tmp_path / "syn"), "--campaigns", "2", "--noise", "1"])
    assert result.exit_code == 0, result.output
    assert any((tmp_path / "syn").iterdir())


def test_models_downloads_to_models_dir(tmp_path, monkeypatch):
    calls = []

    def download_face_models(dest_dir, force=False):
        calls.append((Path(dest_dir), force))
        p = tmp_path / "model.onnx"
        p.write_bytes(b"x" * 10)
        return [p]

    fake_module(monkeypatch, "slad.features.model_download", download_face_models=download_face_models)
    result = runner.invoke(app, ["models", "--force"])
    assert result.exit_code == 0, result.output
    assert calls == [(ROOT / "models", True)] and "model.onnx" in result.output

    def failing(dest_dir, force=False):
        raise RuntimeError("Nie udało się pobrać modeli twarzy: brak sieci")

    fake_module(monkeypatch, "slad.features.model_download", download_face_models=failing)
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 1 and "brak sieci" in result.output


def test_ui_launches_streamlit_with_workspace(tmp_path, monkeypatch):
    fake_root = tmp_path / "repo"
    (fake_root / "slad" / "ui").mkdir(parents=True)
    (fake_root / "slad" / "ui" / "app.py").write_text("# UI", encoding="utf-8")
    monkeypatch.setattr(cli, "ROOT", fake_root)
    seen = {}

    def fake_call(cmd, env=None, cwd=None):
        seen.update(cmd=cmd, env=env, cwd=cwd)
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    ws_dir = tmp_path / "ws"
    result = runner.invoke(app, ["ui", "-w", str(ws_dir), "--port", "8600", "--host", "0.0.0.0", "--no-browser"])
    assert result.exit_code == 0, result.output
    assert seen["env"]["SLAD_WORKSPACE"] == str(ws_dir.resolve())
    assert seen["cmd"][4] == str(fake_root / "slad" / "ui" / "app.py") and "8600" in seen["cmd"]
    assert seen["cwd"] == str(fake_root) and "localhost:8600" in result.output
    assert seen["cmd"][seen["cmd"].index("--server.address") + 1] == "0.0.0.0"

    # domyślnie tylko localhost — bez wystawiania materiałów w sieci lokalnej
    assert runner.invoke(app, ["ui", "-w", str(ws_dir), "--no-browser"]).exit_code == 0
    assert seen["cmd"][seen["cmd"].index("--server.address") + 1] == "localhost"


def test_demo_runs_end_to_end_with_fakes(tmp_path, monkeypatch):
    dataset = tmp_path / "syn" / "dataset.jsonl"
    fake_module(monkeypatch, "slad.ingest.synthetic", generate=lambda out_dir, seed=7: dataset)
    seen = {}

    def fake_process(path, ws, cfg, progress=None, **kwargs):
        seen.update(path=path, offline=not cfg.network.enabled, root=ws.root)
        make_workspace(ws.root)
        return {"materials": 2, "links": 1, "campaigns": 1, "clustered": 2}

    monkeypatch.setattr(pipeline, "process_dataset", fake_process)
    fake_module(monkeypatch, "slad.eval")
    fake_module(monkeypatch, "slad.eval.run", evaluate=lambda ws, cfg: {"ari": 1.0}, format_report=lambda r: "ARI 1.00")
    ws_dir = tmp_path / "demo-ws"
    result = runner.invoke(app, ["demo", "-w", str(ws_dir), "--data", str(tmp_path / "syn"), "--no-llm"])
    assert result.exit_code == 0, result.output
    assert seen == {"path": dataset, "offline": True, "root": ws_dir.resolve()}
    assert "ARI 1.00" in result.output and "ui --workspace" in result.output
    assert (ws_dir / "report.json").exists()
