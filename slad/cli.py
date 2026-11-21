"""Interfejs wiersza poleceń:  python -m slad <komenda>   (również ./start.sh <komenda>, start.bat <komenda>)

demo                       zbiór syntetyczny → przetwarzanie offline → ewaluacja → jak otworzyć UI
run <zbiór>                materiały → cechy → powiązania (z przesłankami) → kampanie
ingest <zbiór>             podgląd wczytania zbioru (format, pola, etykiety) — bez ekstrakcji cech
link                       przeliczenie powiązań i kampanii z zapisanych cech (np. po zmianie wag)
analyze <url|plik|tekst>   nowy materiał → znana kampania / potencjalnie nowa kampania + przesłanki
campaigns [K001]           lista kampanii albo szczegóły jednej
export --format neo4j      Neo4j (CSV + Cypher) · GraphML · JSON
eval                       metryki jakości (gdy materiały mają etykiety kampanii)
synthetic                  syntetyczny zbiór z etykietami
models                     pobranie modeli twarzy (YuNet + SFace)
ui                         interfejs graficzny (Streamlit)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import webbrowser
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NoReturn

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from slad.settings import ROOT, Settings, load_settings

if TYPE_CHECKING:
    from slad.models import Assignment, Campaign, CampaignSummary, FeatureSet, Material
    from slad.store import Workspace

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,  # zmienne lokalne mogą zawierać klucz API LLM
)
console = Console()
log = logging.getLogger("slad.cli")
_STATE = {"verbose": False}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".jfif", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif", ".heic"}
HTML_EXTS = {".html", ".htm", ".xhtml", ".shtml", ".mhtml", ".mht"}
_URL_SCHEME = re.compile(r"^https?://", re.I)
_BARE_URL = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}(?::\d+)?(?:[/?#]\S*)?$", re.I)
_MAX_TEXT_FILE = 200_000

# ------------------------------------------------------------------ wspólne opcje

WorkspaceOpt = Annotated[
    Path | None,
    typer.Option(
        "--workspace",
        "-w",
        help="Katalog roboczy (domyślnie: SLAD_WORKSPACE albo workspace z config/settings.yaml).",
        show_default=False,
    ),
]
OfflineOpt = Annotated[
    bool,
    typer.Option("--offline", help="Bez sieci: bez pobierania stron, DNS, RDAP i TLS (wejście, meta.infra, cache)."),
]
NoLlmOpt = Annotated[bool, typer.Option("--no-llm", help="Bez LLM — same heurystyki (szybciej, powtarzalnie).")]
NoFacesOpt = Annotated[bool, typer.Option("--no-faces", help="Bez wykrywania i rozpoznawania twarzy.")]


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Więcej komunikatów (logi INFO, pełne błędy).")
    ] = False,
) -> None:
    """[bold]Ślad Fejka[/] — od pojedynczych reklam do kampanii oszustw (NASK: Kolekcjoner Fejków).

    Start bez danych NASK: [bold]python -m slad demo[/], potem [bold]python -m slad ui --workspace workspace/demo[/].
    """
    _setup_logging(verbose)


# ------------------------------------------------------------------ przetwarzanie


@app.command()
def run(
    dataset: Annotated[
        Path,
        typer.Argument(
            help="Zbiór: plik CSV / JSON / JSONL / XLSX, katalog (obrazy, HTML, pliki) albo lista URL (.txt).",
            show_default=False,
        ),
    ],
    workspace: WorkspaceOpt = None,
    offline: OfflineOpt = False,
    limit: Annotated[int | None, typer.Option("--limit", "-n", help="Tylko N pierwszych materiałów.")] = None,
    no_llm: NoLlmOpt = False,
    no_faces: NoFacesOpt = False,
    force: Annotated[bool, typer.Option("--force", help="Policz wszystkie cechy od nowa (bez zapisanych).")] = False,
    workers: Annotated[
        int | None, typer.Option("--workers", "-j", help="Wątki ekstrakcji (domyślnie z konfiguracji).")
    ] = None,
) -> None:
    """Pełne przetwarzanie zbioru: materiały → cechy → powiązania (z przesłankami) → kampanie."""
    if not dataset.exists():
        _fail(f"Nie znaleziono zbioru: {dataset}", 2)
    cfg = _settings(workspace, offline=offline, no_llm=no_llm, no_faces=no_faces, workers=workers)
    ws = _workspace(cfg)
    _print_mode(cfg, ws.root)
    with _errors("przetwarzanie"):
        summary = _process_with_progress(dataset, ws, cfg, limit=limit, force=force)
    _print_run_summary(summary)
    if summary.get("link_error"):
        raise typer.Exit(1)
    if summary.get("materials"):
        hint = f" --workspace {_display_path(ws.root)}" if workspace else ""
        console.print(f"Dalej: [bold]python -m slad campaigns{hint}[/] · [bold]python -m slad ui{hint}[/]")


@app.command()
def ingest(
    dataset: Annotated[Path, typer.Argument(help="Zbiór do sprawdzenia.", show_default=False)],
    workspace: WorkspaceOpt = None,
    limit: Annotated[int | None, typer.Option("--limit", "-n", help="Tylko N pierwszych materiałów.")] = None,
    show: Annotated[int, typer.Option("--show", help="Ile przykładowych materiałów pokazać.")] = 10,
    save: Annotated[bool, typer.Option("--save", help="Zapisz materiały w workspace (bez cech).")] = False,
) -> None:
    """Podgląd wczytania zbioru: rozpoznane pola, kompletność, etykiety — bez ekstrakcji cech."""
    if not dataset.exists():
        _fail(f"Nie znaleziono zbioru: {dataset}", 2)
    with _errors("wczytywanie"), console.status("Wczytywanie zbioru…"):
        from slad.ingest import load_dataset

        materials = list(load_dataset(dataset))
    if limit is not None:
        materials = materials[: max(0, limit)]
    _print_dataset_overview(materials, show)
    if save and materials:
        ws = _workspace(_settings(workspace))
        ws.save_materials(materials)
        console.print(f"✓ Zapisano {len(materials)} materiałów w {_display_path(ws.root)}")


@app.command()
def link(workspace: WorkspaceOpt = None) -> None:
    """Przelicza powiązania i kampanie z zapisanych cech (np. po zmianie wag w config/settings.yaml)."""
    cfg = _settings(workspace)
    ws = _workspace(cfg)
    if not ws.materials():
        _fail(f"Pusty workspace {_display_path(ws.root)} — najpierw: python -m slad run <zbiór>")
    with _errors("łączenie"), _progress_bar() as progress:
        from slad.pipeline import rebuild_links

        task = progress.add_task("Łączenie materiałów w kampanie…", total=None)
        summary = rebuild_links(ws, cfg, stage_progress=_stage_updater(progress, task))
    _print_run_summary(summary)


@app.command()
def analyze(
    target: Annotated[
        str,
        typer.Argument(
            help="URL, ścieżka do zrzutu ekranu / zapisanego HTML / pliku tekstowego albo sam tekst reklamy.",
            show_default=False,
        ),
    ],
    workspace: WorkspaceOpt = None,
    offline: OfflineOpt = False,
    no_llm: NoLlmOpt = False,
    no_faces: NoFacesOpt = False,
    kind: Annotated[str | None, typer.Option("--as", help="Wymuś typ wejścia: url | image | html | text.")] = None,
    add: Annotated[bool, typer.Option("--add", help="Dodaj materiał do bazy i przelicz kampanie.")] = False,
    top: Annotated[int, typer.Option("--top", help="Ile najbliższych materiałów pokazać.")] = 5,
) -> None:
    """Nowy materiał → znana kampania albo sygnał nowej kampanii, z przesłankami powiązań."""
    try:
        inputs = detect_input(target, kind)
    except ValueError as e:
        _fail(str(e), 2)
    cfg = _settings(workspace, offline=offline, no_llm=no_llm, no_faces=no_faces)
    ws = _workspace(cfg)
    if not ws.materials():
        console.print(f"[yellow]Pusta baza {_display_path(ws.root)}[/] — materiał nie ma z czym się porównać.")
    with _errors("analiza"), console.status("Analiza materiału…"):
        from slad.ingest import material_from_input
        from slad.pipeline import analyze_new

        material = material_from_input(**inputs)
        fs, assignment = analyze_new(material, ws, cfg, add=add)
    _print_assignment(material, fs, assignment, ws, top)
    if add:
        console.print(f"✓ Dodano do bazy {_display_path(ws.root)} (kampanie przeliczone)")


# ------------------------------------------------------------------ wyniki


@app.command()
def campaigns(
    campaign_id: Annotated[
        str | None, typer.Argument(help="Id kampanii (np. K001) — szczegóły.", show_default=False)
    ] = None,
    workspace: WorkspaceOpt = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Ile kampanii pokazać.")] = 50,
) -> None:
    """Lista kampanii (największe pierwsze) albo szczegóły jednej: przesłanki, osoby, domeny, materiały."""
    ws = _workspace(_settings(workspace))
    items = ws.campaigns()
    if not items:
        console.print(f"[yellow]Brak kampanii w {_display_path(ws.root)}[/] — najpierw: python -m slad run <zbiór>")
        return
    if campaign_id:
        found = next((c for c in items if c.id.lower() == campaign_id.strip().lower()), None)
        if found is None:
            _fail(f"Nie ma kampanii {campaign_id} (dostępne: {', '.join(c.id for c in items[:20])}…)", 2)
        _print_campaign(ws, found)
        return
    _print_campaigns(items, limit, n_materials=len(ws.materials()))


@app.command()
def export(
    workspace: WorkspaceOpt = None,
    fmt: Annotated[str, typer.Option("--format", "-f", help="neo4j | graphml | json | all")] = "neo4j",
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Katalog wynikowy (domyślnie <workspace>/export).", show_default=False),
    ] = None,
) -> None:
    """Eksport grafu: Neo4j (CSV + import.cypher), GraphML (Gephi, yEd), JSON (node-link)."""
    fmt = fmt.strip().lower()
    if fmt not in {"neo4j", "graphml", "json", "all"}:
        _fail(f"Nieznany format: {fmt} (dozwolone: neo4j, graphml, json, all)", 2)
    ws = _workspace(_settings(workspace))
    if not ws.materials():
        _fail(f"Pusty workspace {_display_path(ws.root)} — najpierw: python -m slad run <zbiór>")
    target = out.expanduser().resolve() if out else ws.root / "export"
    out_dir = target.parent if target.suffix.lower() in {".graphml", ".json"} else target
    written: list[Path] = []
    with _errors("eksport"):
        from slad.export import export_graphml, export_json, export_neo4j

        if fmt in {"neo4j", "all"}:
            written += export_neo4j(ws, out_dir)
        if fmt in {"graphml", "all"}:
            file = target if target.suffix.lower() == ".graphml" else out_dir / "slad_fejka.graphml"
            written.append(export_graphml(ws, file))
        if fmt in {"json", "all"}:
            file = target if target.suffix.lower() == ".json" else out_dir / "slad_fejka.json"
            written.append(export_json(ws, file))
    for p in written:
        console.print(f"  ✓ {_display_path(p)}")
    if fmt in {"neo4j", "all"}:
        console.print(
            "Neo4j: skopiuj pliki CSV do katalogu [bold]import/[/] bazy i uruchom "
            f"[bold]cypher-shell -f {_display_path(out_dir / 'import.cypher')}[/] (instrukcja w nagłówku pliku)."
        )


@app.command("eval")
def eval_(workspace: WorkspaceOpt = None) -> None:
    """Metryki jakości grupowania i przypisań (wymaga etykiet kampanii, np. w zbiorze syntetycznym)."""
    cfg = _settings(workspace)
    ws = _workspace(cfg)
    if not ws.materials():
        _fail(f"Pusty workspace {_display_path(ws.root)} — najpierw: python -m slad run <zbiór>")
    with _errors("ewaluacja"), console.status("Ewaluacja…"):
        report, text, path = _evaluate(ws, cfg)
    _print_report(text)
    console.print(f"✓ Raport JSON: {_display_path(path)}")


# ------------------------------------------------------------------ narzędzia


@app.command()
def synthetic(
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Katalog wynikowy (domyślnie data/synthetic).")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Ziarno losowania (powtarzalny zbiór).")] = 7,
    n_campaigns: Annotated[int, typer.Option("--campaigns", help="Liczba kampanii.")] = 8,
    noise: Annotated[int, typer.Option("--noise", help="Liczba materiałów spoza kampanii (szum).")] = 12,
) -> None:
    """Generuje syntetyczny zbiór z etykietami kampanii (testy i demo bez danych NASK)."""
    out_dir = out.expanduser().resolve() if out else ROOT / "data" / "synthetic"
    with _errors("generowanie zbioru"), console.status("Generowanie zbioru syntetycznego…"):
        from slad.ingest.synthetic import generate

        path = generate(out_dir, seed=seed, n_campaigns=n_campaigns, noise=noise)
    console.print(f"✓ Zbiór syntetyczny: {_display_path(Path(path))}")
    console.print(f"Dalej: [bold]python -m slad run {_display_path(Path(path))} --offline[/]")


@app.command()
def models(
    force: Annotated[bool, typer.Option("--force", help="Pobierz ponownie, nawet gdy pliki są poprawne.")] = False,
) -> None:
    """Pobiera modele twarzy (YuNet + SFace z OpenCV Zoo, ≈40 MB) do katalogu models/."""
    cfg = load_settings()
    dest = cfg.path(cfg.faces.detector_model).parent
    with _errors("pobieranie modeli"), console.status("Pobieranie modeli twarzy…"):
        from slad.features.model_download import download_face_models

        paths = download_face_models(dest, force=force)
    for p in paths:
        p = Path(p)
        size = f"{p.stat().st_size / 1e6:.1f} MB" if p.exists() else "?"
        console.print(f"  ✓ {_display_path(p)} ({size})")


@app.command()
def ui(
    workspace: WorkspaceOpt = None,
    port: Annotated[int, typer.Option("--port", "-p", help="Port HTTP.")] = 8501,
    host: Annotated[
        str,
        typer.Option("--host", help="Adres nasłuchu: localhost = tylko ten komputer; 0.0.0.0 = sieć lokalna / Docker."),
    ] = "localhost",
    browser: Annotated[bool, typer.Option("--browser/--no-browser", help="Otwórz przeglądarkę.")] = True,
) -> None:
    """Interfejs graficzny (Streamlit): przegląd, kampanie, graf, przesłanki, analiza nowego materiału.

    Domyślnie tylko localhost — materiały NASK nie są wystawiane w sieci lokalnej bez --host 0.0.0.0.
    """
    cfg = _settings(workspace)
    ws_root = cfg.path(cfg.workspace)
    app_file = ROOT / "slad" / "ui" / "app.py"
    if not app_file.exists():
        _fail(f"Brak pliku interfejsu: {app_file}")
    if not (ws_root / "materials.jsonl").exists():
        console.print(
            f"[yellow]Workspace {_display_path(ws_root)} jest pusty[/] — dane: "
            "[bold]python -m slad demo[/] albo [bold]python -m slad run <zbiór>[/]"
        )
    url = f"http://{'localhost' if host in (None, '', '0.0.0.0', '127.0.0.1', '::') else host}:{port}"
    console.print(f"Interfejs: [bold]{url}[/] · workspace: {_display_path(ws_root)} · Ctrl+C kończy")
    if browser:
        _open_browser_later(url)
    env = {**os.environ, "SLAD_WORKSPACE": str(ws_root)}
    try:
        code = subprocess.call(streamlit_command(app_file, port, host), env=env, cwd=str(ROOT))
    except KeyboardInterrupt:
        code = 0
    raise typer.Exit(code)


@app.command()
def demo(
    workspace: WorkspaceOpt = None,
    data: Annotated[
        Path | None, typer.Option("--data", help="Katalog na zbiór syntetyczny (domyślnie data/synthetic).")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Ziarno zbioru syntetycznego.")] = 7,
    llm: Annotated[
        bool,
        typer.Option(
            "--llm/--no-llm",
            help="LLM do cech tekstowych. W demo domyślnie wyłączony: lokalny model to ~13 s na materiał "
            "(73 materiały ≈ 16 min), heurystyki — 10 s na cały zbiór.",
        ),
    ] = False,
    no_faces: NoFacesOpt = False,
    open_ui: Annotated[bool, typer.Option("--ui", help="Na koniec uruchom interfejs.")] = False,
) -> None:
    """Demonstracja end-to-end bez danych NASK: zbiór syntetyczny → przetwarzanie offline → ewaluacja."""
    cfg = _settings(workspace or ROOT / "workspace" / "demo", offline=True, no_llm=not llm, no_faces=no_faces)
    ws = _workspace(cfg)
    data_dir = data.expanduser().resolve() if data else ROOT / "data" / "synthetic"

    console.rule("[bold]1/3 · Zbiór syntetyczny z etykietami")
    with _errors("generowanie zbioru"), console.status("Generowanie…"):
        from slad.ingest.synthetic import generate

        dataset = Path(generate(data_dir, seed=seed))
    console.print(f"✓ {_display_path(dataset)}")

    console.rule("[bold]2/3 · Przetwarzanie (offline)")
    _print_mode(cfg, ws.root)
    with _errors("przetwarzanie"):
        summary = _process_with_progress(dataset, ws, cfg)
    _print_run_summary(summary)

    console.rule("[bold]3/3 · Ewaluacja")
    try:
        with console.status("Ewaluacja…"):
            _, text, path = _evaluate(ws, cfg)
        _print_report(text)
        console.print(f"✓ Raport JSON: {_display_path(path)}")
    except Exception as e:  # noqa: BLE001 — demo bez ewaluacji nadal ma sens
        console.print(f"[yellow]Ewaluacja niedostępna:[/] {escape(str(e))}")

    rel = _display_path(ws.root)
    console.print(
        Panel(
            f"Interfejs:  [bold]./start.sh ui --workspace {rel}[/]\n"
            f"Windows:    [bold]start.bat ui --workspace {rel}[/]\n"
            f"lub:        [bold]python -m slad ui --workspace {rel}[/]  →  http://localhost:8501\n"
            f"Kampanie w terminalu: python -m slad campaigns --workspace {rel}",
            title="Gotowe",
            expand=False,
        )
    )
    if open_ui:
        ui(workspace=ws.root, port=8501, host="localhost", browser=True)


# ------------------------------------------------------------------ publiczne pomocnicze


def detect_input(target: str, kind: str | None = None) -> dict[str, str]:
    """Typ wejścia `analyze`: URL, zrzut ekranu, zapisany HTML, plik tekstowy albo sam tekst.

    Zwraca argumenty dla slad.ingest.material_from_input (url / image_path / html_path / text).
    Nieprawidłowe wejście → ValueError z komunikatem po polsku.
    """
    raw = target.strip()
    if not raw:
        raise ValueError("Puste wejście — podaj URL, ścieżkę do pliku albo tekst.")
    kind = (kind or "").strip().lower() or None
    if kind is not None and kind not in {"url", "image", "html", "text"}:
        raise ValueError(f"Nieznany typ wejścia: {kind} (dozwolone: url, image, html, text)")
    path = Path(raw).expanduser()
    is_file = _is_file(raw, path)

    if kind == "url" or (kind is None and _URL_SCHEME.match(raw)):
        return {"url": raw if _URL_SCHEME.match(raw) else f"http://{raw}"}
    if kind in {"image", "html"}:
        if not is_file:
            raise ValueError(f"Nie znaleziono pliku: {raw}")
        return {f"{kind}_path": str(path.resolve())}
    if kind == "text":
        return {"text": _read_text(path) if is_file else raw}

    if is_file:
        return _detect_file(path)
    if _is_dir(raw, path):
        raise ValueError(f"{raw} to katalog — cały zbiór przetwarza: python -m slad run {raw}")
    if _BARE_URL.match(raw):
        return {"url": f"http://{raw}"}
    return {"text": raw}


def streamlit_command(app_file: Path, port: int = 8501, host: str | None = None) -> list[str]:
    """Polecenie uruchamiające UI (bez pytania o e-mail i bez telemetrii Streamlit)."""
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_file),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]  # fmt: skip
    if host:
        cmd += ["--server.address", host]
    return cmd


# ------------------------------------------------------------------ wewnętrzne


def _setup_logging(verbose: bool) -> None:
    _STATE["verbose"] = verbose
    level = (os.environ.get("LOG_LEVEL") or ("INFO" if verbose else "WARNING")).upper()
    root = logging.getLogger()
    if not any(isinstance(h, RichHandler) for h in root.handlers):
        handler = RichHandler(console=console, show_time=False, show_path=False, markup=False)
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        root.addHandler(handler)
    root.setLevel(level if level in logging.getLevelNamesMapping() else logging.WARNING)


def _fail(message: str, code: int = 1) -> NoReturn:
    console.print(f"[red]✗ {escape(message)}[/]")
    raise typer.Exit(code)


@contextmanager
def _errors(action: str) -> Iterator[None]:
    """Czytelny komunikat zamiast śladu stosu (pełny ślad: --verbose)."""
    try:
        yield
    except (typer.Exit, typer.Abort):
        raise
    except KeyboardInterrupt:
        console.print("[yellow]Przerwano.[/] Policzone cechy są zapisane — ponowne uruchomienie je wykorzysta.")
        raise typer.Exit(130) from None
    except Exception as e:  # noqa: BLE001
        if _STATE["verbose"]:
            console.print_exception(show_locals=False)
        _fail(f"Błąd ({action}): {type(e).__name__}: {e}")


def _settings(
    workspace: Path | str | None = None,
    *,
    offline: bool = False,
    no_llm: bool = False,
    no_faces: bool = False,
    workers: int | None = None,
) -> Settings:
    """load_settings + nadpisania z CLI (pola zagnieżdżone ustawiane po wczytaniu, by nie gubić reszty sekcji)."""
    overrides: dict[str, Any] = {}
    if workspace:
        overrides["workspace"] = str(Path(workspace).expanduser().resolve())
    cfg = load_settings(**overrides)
    if offline:
        cfg.network.enabled = False
    if no_llm:
        cfg.llm.enabled = False
    if no_faces:
        cfg.faces.enabled = False
    if workers:
        cfg.workers = max(1, workers)
    return cfg


def _workspace(cfg: Settings) -> Workspace:
    from slad.store import Workspace

    return Workspace(cfg.path(cfg.workspace))


def _display_path(p: Path) -> str:
    """Ścieżka względem bieżącego katalogu (krócej w komunikatach), w razie potrzeby absolutna."""
    p = Path(p)
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix() or "."
    except ValueError:
        return str(p)


def _progress_bar() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def _stage_updater(progress: Progress, task: Any) -> Any:
    """stage_progress(etap, zrobione, wszystkie) z slad.link.build → ten sam pasek postępu."""

    def on_stage(stage: str, done: int, total: int) -> None:
        progress.update(task, completed=done, total=max(total, 1), description=escape(f"Łączenie · {stage}"))

    return on_stage


def _process_with_progress(dataset: Path, ws: Workspace, cfg: Settings, **kwargs: Any) -> dict[str, Any]:
    from slad.pipeline import process_dataset

    with _progress_bar() as progress:
        task = progress.add_task("Wczytywanie zbioru…", total=None)

        def on_progress(done: int, total: int, material_id: str) -> None:
            if done >= total:
                desc = "Łączenie materiałów w kampanie…"
            else:
                desc = f"Ekstrakcja cech · {material_id[:40]}" if material_id else "Ekstrakcja cech…"
            progress.update(task, completed=done, total=total, description=escape(desc))

        return process_dataset(
            dataset, ws, cfg, progress=on_progress, stage_progress=_stage_updater(progress, task), **kwargs
        )


def _evaluate(ws: Workspace, cfg: Settings) -> tuple[dict[str, Any], Any, Path]:
    from slad.eval.run import evaluate, format_report

    report = evaluate(ws, cfg)
    path = ws.path("report.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report, format_report(report), path


def _print_report(text: Any) -> None:
    if isinstance(text, str):
        console.print(text, markup=False, highlight=False)
    elif text is not None:
        console.print(text)


def _print_mode(cfg: Settings, ws_root: Path) -> None:
    net = "[green]online[/]" if cfg.network.enabled else "[yellow]offline[/]"
    if cfg.llm.enabled and (cfg.llm.base_url or cfg.llm.api_key):
        llm = f"[green]{escape(cfg.llm.model or 'tak')}[/]"
    else:
        llm = "heurystyki"
    console.print(
        f"Sieć: {net} · LLM: {llm} · twarze: {'tak' if cfg.faces.enabled else 'nie'} · "
        f"OCR: {'tak' if cfg.ocr.enabled else 'nie'} · wątki: {cfg.workers} · "
        f"workspace: [bold]{escape(_display_path(ws_root))}[/]"
    )


def _print_run_summary(s: dict[str, Any]) -> None:
    t = Table(show_header=False, box=None, pad_edge=False)
    t.add_column(style="bold")
    t.add_column()
    if "materials" in s:
        dup = f" [dim](pominięte powtórzone id: {s['duplicate_ids']})[/]" if s.get("duplicate_ids") else ""
        t.add_row("Materiały", f"{s['materials']}{dup}")
    if "features" in s:
        llm = f" · z LLM: {s['llm_used']}" if s.get("llm_used") else ""
        t.add_row("Cechy", f"policzone {s.get('extracted', 0)} · z zapisu {s.get('cached', 0)}{llm}")
        available = s.get("available") or {}
        if available:
            groups = " · ".join(f"{k} {v}" for k, v in sorted(available.items(), key=lambda kv: -kv[1]))
            t.add_row("Grupy cech", groups)
    relations = " · ".join(f"{k} {v}" for k, v in (s.get("relations") or {}).items())
    t.add_row("Powiązania", f"{s.get('links', 0)}" + (f" ({relations})" if relations else ""))
    largest = f" · największa: {s['largest_campaign']}" if s.get("largest_campaign") else ""
    t.add_row("Kampanie", f"{s.get('campaigns', 0)} · materiały w kampaniach: {s.get('clustered', 0)}{largest}")
    if "errors" in s:
        style = "red" if s["errors"] else "green"
        t.add_row("Materiały z błędami", f"[{style}]{s['errors']}[/]")
    times = [f"{label} {s[key]:.1f} s" for key, label in (("extract_s", "do cech"), ("link_s", "łączenie")) if key in s]
    if "time_s" in s:
        t.add_row("Czas", f"{s['time_s']:.1f} s" + (f" ({', '.join(times)})" if times else ""))
    elif times:
        t.add_row("Czas", ", ".join(times))
    console.print(Panel(t, title="Podsumowanie", expand=False))
    for line in (s.get("error_samples") or [])[:5]:
        console.print(f"  [dim]• {escape(line)}[/]")
    if s.get("link_error"):
        console.print(
            f"[red]✗ Łączenie nie powiodło się:[/] {escape(s['link_error'])} — po poprawce: python -m slad link"
        )
    elif "materials" in s and not s["materials"]:
        console.print("[yellow]Zbiór nie zawiera materiałów[/] — sprawdź format: python -m slad ingest <zbiór>")


def _top(pairs: list[tuple[str, int]] | list[Any], n: int) -> str:
    out = []
    for item in pairs[:n]:
        value, count = (item[0], item[1]) if isinstance(item, list | tuple) and len(item) == 2 else (item, None)
        out.append(f"{value} ({count})" if count else str(value))
    return ", ".join(out)


def _period(s: CampaignSummary) -> str:
    if not s.first_seen and not s.last_seen:
        return "—"
    first = s.first_seen.date().isoformat() if s.first_seen else "?"
    last = s.last_seen.date().isoformat() if s.last_seen else "?"
    return first if first == last else f"{first} – {last}"


def _print_campaigns(items: list[Campaign], limit: int, n_materials: int) -> None:
    clustered = len({m for c in items for m in c.members})
    t = Table(title=f"Kampanie: {len(items)} · materiały w kampaniach: {clustered}/{n_materials}")
    t.add_column("ID", style="bold", no_wrap=True)
    t.add_column("Nazwa")
    t.add_column("Mat.", justify="right")
    t.add_column("Spójność", justify="right")
    t.add_column("Osoby / marki")
    t.add_column("Narracje")
    t.add_column("Domeny")
    t.add_column("Okres", no_wrap=True)
    for c in items[: max(1, limit)]:
        s = c.summary
        who = _top(s.persons, 2) or _top(s.brands, 2)
        t.add_row(
            c.id,
            escape(c.name),
            str(len(c.members)),
            f"{c.cohesion:.1f}",
            escape(who) or "—",
            escape(_top(s.narratives, 2)) or "—",
            escape(_top(s.domains, 2)) or "—",
            _period(s),
        )
    console.print(t)
    if len(items) > limit:
        console.print(f"[dim]… i {len(items) - limit} kolejnych (--limit)[/]")
    console.print("[dim]Szczegóły: python -m slad campaigns <ID>[/]")


def _print_campaign(ws: Workspace, c: Campaign) -> None:
    s = c.summary
    lines = [
        f"[bold]{escape(c.name)}[/]",
        f"Materiały: {len(c.members)} · spójność: {c.cohesion:.2f} · okres: {_period(s)}",
    ]
    for label, pairs in (
        ("Osoby", s.persons),
        ("Marki", s.brands),
        ("Narracje", s.narratives),
        ("Domeny", s.domains),
        ("Wskaźniki", s.indicators),
    ):
        if pairs:
            lines.append(f"{label}: {escape(_top(pairs, 8))}")
    if s.keywords:
        lines.append(f"Słowa kluczowe: {escape(', '.join(s.keywords[:12]))}")
    console.print(Panel("\n".join(lines), title=f"Kampania {c.id}", expand=False))
    if c.key_evidence:
        t = Table(title="Kluczowe przesłanki")
        t.add_column("Przesłanka")
        t.add_column("Grupa")
        t.add_column("Waga", justify="right")
        for e in c.key_evidence[:10]:
            t.add_row(escape(e.detail), e.group, f"{e.weight:.1f}")
        console.print(t)
    t = Table(title="Materiały")
    t.add_column("ID", no_wrap=True)
    t.add_column("Źródło")
    t.add_column("Platforma")
    t.add_column("Adres / treść")
    for mid in c.members[:30]:
        m = ws.material(mid)
        if m is None:
            t.add_row(escape(mid), "?", "", "")
            continue
        t.add_row(escape(m.id), m.source.value, escape(m.platform or ""), escape(_material_hint(m)))
    console.print(t)
    if len(c.members) > 30:
        console.print(f"[dim]… i {len(c.members) - 30} kolejnych materiałów[/]")


def _material_hint(m: Material, width: int = 70) -> str:
    text = m.url or (m.text or "").replace("\n", " ") or (Path(m.image_paths[0]).name if m.image_paths else "") or ""
    return text if len(text) <= width else text[: width - 1] + "…"


def _print_assignment(material: Material, fs: FeatureSet, a: Assignment, ws: Workspace, top: int) -> None:
    from slad.models import AssignmentStatus

    known = {c.id: c for c in ws.campaigns()}
    if a.status == AssignmentStatus.ASSIGNED:
        c = known.get(a.campaign_id or "")
        head = f"[bold green]Przypisano do kampanii {escape(a.campaign_id or '?')}[/]"
        if c:
            head += f" — {escape(c.name)} ({len(c.members)} mat.)"
    elif a.status == AssignmentStatus.NEW_CAMPAIGN:
        head = "[bold yellow]Potencjalnie NOWA kampania[/] — powiązania z materiałami spoza znanych kampanii"
    else:
        head = "[bold]Brak powiązań[/] — nic podobnego w bazie"
    body = f"{head}\nWynik: {a.score:.1f}"
    if a.explanation:
        body += f"\n\n{escape(a.explanation)}"
    console.print(Panel(body, title=f"Materiał {escape(material.id)}", expand=False))

    facts = [f"Cechy: {', '.join(fs.available) or 'brak'}"]
    tx = fs.text
    for label, values in (("osoby", tx.persons), ("marki", tx.brands), ("narracje", tx.narratives)):
        if values:
            facts.append(f"{label}: {', '.join(values[:5])}")
    if fs.infra.domains:
        facts.append(f"domeny: {', '.join(fs.infra.domains[:5])}")
    iocs = sum(len(v) for v in tx.iocs.values())
    if iocs:
        facts.append(f"wskaźniki IOC: {iocs}")
    console.print(escape(" · ".join(facts)))
    for err in fs.errors[:5]:
        console.print(f"  [dim]• {escape(err)}[/]")

    if not a.nearest:
        return
    campaign_of = {m: c.id for c in known.values() for m in c.members}
    t = Table(title="Najbliższe materiały i przesłanki powiązań", show_lines=True)
    t.add_column("Materiał")
    t.add_column("Kampania", no_wrap=True)
    t.add_column("Wynik", justify="right")
    t.add_column("Relacja")
    t.add_column("Przesłanki (waga)")
    for lk in a.nearest[: max(1, top)]:
        other = lk.b if lk.a == a.material_id else lk.a if lk.b == a.material_id else f"{lk.a} ↔ {lk.b}"
        m = ws.material(other)
        who = escape(other) + (f"\n[dim]{escape(_material_hint(m, 40))}[/]" if m else "")
        evidence = sorted(lk.evidence, key=lambda e: e.weight, reverse=True)
        lines = [f"• {escape(e.detail)} [dim]({e.weight:+.1f})[/]" for e in evidence[:6]]
        if len(evidence) > 6:
            lines.append(f"[dim]… i {len(evidence) - 6} więcej[/]")
        t.add_row(who, campaign_of.get(other, "—"), f"{lk.score:.1f}", lk.relation.value, "\n".join(lines) or "—")
    console.print(t)


def _print_dataset_overview(materials: list[Material], show: int) -> None:
    n = len(materials)
    if not n:
        console.print(
            "[yellow]Nie wczytano żadnego materiału[/] — sprawdź format i nazwy kolumn (docs/ARCHITEKTURA.md)."
        )
        return
    sources = Counter(m.source.value for m in materials)
    platforms = Counter(m.platform or "—" for m in materials)
    labels = Counter(m.label for m in materials if m.label)
    missing = sum(1 for m in materials for p in m.image_paths if not Path(p).exists())
    t = Table(title=f"Wczytano {n} materiałów", show_header=False)
    t.add_column(style="bold")
    t.add_column()
    t.add_row("Źródła", ", ".join(f"{k} {v}" for k, v in sources.most_common()))
    t.add_row("Platformy", escape(", ".join(f"{k} {v}" for k, v in platforms.most_common(8))))
    for label, count in (
        ("z adresem URL", sum(1 for m in materials if m.url)),
        ("z tekstem", sum(1 for m in materials if m.text)),
        ("z obrazami", sum(1 for m in materials if m.image_paths)),
        ("z zapisanym HTML", sum(1 for m in materials if m.html_path)),
        ("z datą", sum(1 for m in materials if m.observed_at)),
        ("z reklamodawcą", sum(1 for m in materials if m.advertiser)),
        ("z gotową infrastrukturą (meta.infra)", sum(1 for m in materials if m.meta.get("infra"))),
    ):
        t.add_row(label, f"{count} ({count / n:.0%})")
    t.add_row("etykiety kampanii (ewaluacja)", f"{sum(labels.values())} materiałów, {len(labels)} kampanii")
    if missing:
        t.add_row("brakujące pliki obrazów", f"[red]{missing}[/]")
    console.print(t)
    if show > 0:
        sample = Table(title=f"Pierwsze {min(show, n)}")
        for col in ("ID", "Źródło", "Platforma", "Adres / treść", "Obrazy", "Etykieta"):
            sample.add_column(col)
        for m in materials[:show]:
            sample.add_row(
                escape(m.id),
                m.source.value,
                escape(m.platform or ""),
                escape(_material_hint(m, 50)),
                str(len(m.image_paths)),
                escape(m.label or ""),
            )
        console.print(sample)


def _open_browser_later(url: str, delay: float = 2.5) -> None:
    timer = threading.Timer(delay, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()


def _is_file(raw: str, path: Path) -> bool:
    if len(raw) > 1024 or "\n" in raw:
        return False
    try:
        return path.is_file()
    except (OSError, ValueError):
        return False


def _is_dir(raw: str, path: Path) -> bool:
    if len(raw) > 1024 or "\n" in raw:
        return False
    try:
        return path.is_dir()
    except (OSError, ValueError):
        return False


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")[:_MAX_TEXT_FILE]


def _detect_file(path: Path) -> dict[str, str]:
    suffix = path.suffix.lower()
    resolved = str(path.resolve())
    if suffix in IMAGE_EXTS:
        return {"image_path": resolved}
    if suffix in HTML_EXTS:
        return {"html_path": resolved}
    with path.open("rb") as f:
        head = f.read(4096).lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html")) or b"<html" in head[:1024]:
        return {"html_path": resolved}
    if _looks_like_image(path):
        return {"image_path": resolved}
    return {"text": _read_text(path)}


def _looks_like_image(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001
        return False
