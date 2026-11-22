"""Bloodhound — interfejs graficzny (Streamlit).

Uruchomienie z katalogu repozytorium:   streamlit run slad/ui/app.py
Katalog roboczy: zmienna SLAD_WORKSPACE albo ``workspace`` z config/settings.yaml; w panelu bocznym
można wybrać inny podkatalog ``workspace/`` lub wskazać dowolną ścieżkę.

Moduły przetwarzania (pipeline, ingest, export, eval) są importowane leniwie — brak któregoś z nich
wyłącza tylko odpowiednią funkcję, a nie cały interfejs.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import html as _html
import inspect
import json
import logging
import math
import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from slad.models import Assignment, Campaign, Evidence, FeatureSet, Link, Material  # noqa: E402
from slad.settings import ROOT, Settings, load_settings  # noqa: E402
from slad.ui import components as C  # noqa: E402
from slad.ui import graph as G  # noqa: E402

log = logging.getLogger("slad.ui")

# Stan bieżącego przebiegu skryptu (Streamlit wykonuje skrypt od nowa przy każdej interakcji).
_STATE: dict[str, Any] = {}

_CSS = """
<style>
  .block-container { padding-top: 2.4rem; padding-bottom: 3rem; }
  [data-testid="stMetricValue"] { font-size: 1.6rem; }
  .slad-bar { display: flex; height: 12px; border-radius: 6px; overflow: hidden; gap: 2px; margin: 6px 0 6px; }
  .slad-bar > div { height: 100%%; min-width: 3px; }
  .slad-labels { font-size: 0.86rem; color: %(ink2)s; display: flex; flex-wrap: wrap; gap: 4px 14px; align-items: center; }
  .slad-sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; }
  .slad-head { font-size: 1.4rem; font-weight: 700; margin: 0.3rem 0 0.1rem; color: %(ink)s; }
  .slad-head .glyph { font-size: 1.2rem; margin-right: 6px; }
  .slad-muted { color: %(muted)s; font-size: 0.86rem; }
</style>
"""

_ASSETS = Path(__file__).resolve().parent / "assets"
_AVATAR = _ASSETS / "bloodhound_64.png"  # avatar projektu (logo w panelu, ikona karty przeglądarki)


@functools.lru_cache(maxsize=1)
def _logo() -> str:
    """Logo w panelu bocznym: avatar (osadzony jako data URI) + nazwa projektu."""
    try:
        img = base64.b64encode(_AVATAR.read_bytes()).decode("ascii")
        avatar = f'<image x="0" y="0" width="40" height="40" href="data:image/png;base64,{img}"/>'
    except OSError:
        avatar = ""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="40" viewBox="0 0 200 40">'
        f"{avatar}"
        '<text x="50" y="27" font-family="system-ui, -apple-system, Segoe UI, sans-serif" font-size="20" '
        'font-weight="700" fill="#0b0b0b">Bloodhound</text></svg>'
    )


def _page_icon() -> Any:
    """Ikona karty przeglądarki: avatar, a gdy pliku brak — emoji."""
    try:
        from PIL import Image

        return Image.open(_AVATAR)
    except Exception:  # noqa: BLE001 — ikona jest ozdobą
        return "🐶"


# ---------------------------------------------------------------- infrastruktura aplikacji


def _cfg() -> Settings:
    return _STATE["cfg"]


def _b() -> C.Bundle:
    return _STATE["bundle"]


def _root() -> Path:
    return _STATE["root"]


def _load_cfg() -> Settings:
    try:
        return load_settings()
    except Exception as e:  # noqa: BLE001 — uszkodzona konfiguracja: wartości domyślne
        log.exception("Nie udało się wczytać ustawień")
        st.sidebar.error(f"Błąd konfiguracji, używam wartości domyślnych: {e}")
        return Settings()


@st.cache_resource(max_entries=6, show_spinner="Wczytywanie katalogu roboczego…")
def _bundle_cached(root: str, sig: tuple) -> C.Bundle:
    return C.load_bundle(root)


def _load_bundle(root: Path) -> C.Bundle:
    try:
        return _bundle_cached(str(root), C.workspace_signature(root))
    except Exception as e:  # noqa: BLE001
        log.exception("Nie udało się wczytać katalogu roboczego %s", root)
        st.sidebar.error(f"Nie udało się wczytać katalogu roboczego: {e}")
        return C.Bundle(root=root)


@st.cache_data(max_entries=4000, show_spinner=False)
def _thumb_cached(path: str, mtime: int, max_side: int) -> bytes | None:
    return C.thumbnail_bytes(path, max_side)


@st.cache_data(max_entries=500, show_spinner=False)
def _faces_cached(path: str, mtime: int, faces_json: str) -> bytes | None:
    from slad.models import Face

    faces = [Face.model_validate(f) for f in json.loads(faces_json)]
    return C.faces_overlay(path, faces)


@st.cache_resource(max_entries=4, show_spinner="Ładowanie modeli (OCR, twarze, LLM)…")
def _extract_ctx_cached(root: str, cfg_json: str) -> Any:
    from slad.features import ExtractContext

    return ExtractContext.from_settings(Settings.model_validate_json(cfg_json), Path(root))


def _extract_ctx(root: str, cfg_json: str) -> Any:
    """Jeden kontekst ekstrakcji (modele twarzy, klient LLM) na katalog i ustawienia — bez ładowania
    modeli przy każdej analizie. Błąd → None (pipeline utworzy własny kontekst)."""
    try:
        return _extract_ctx_cached(root, cfg_json)
    except Exception:  # noqa: BLE001
        log.warning("Nie udało się przygotować kontekstu ekstrakcji", exc_info=True)
        return None


def _mtime(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _thumb(path: Path | None, max_side: int = 360) -> bytes | None:
    mtime = _mtime(path)
    return _thumb_cached(str(path), mtime, max_side) if mtime is not None else None


def _image_or_placeholder(path: Path | None, max_side: int = 360, caption: str | None = None) -> None:
    data = _thumb(path, max_side)
    if data:
        st.image(data, width="stretch", caption=caption)
    else:
        st.caption("🖼️ brak obrazu")


def _guard(fn: Callable[[], None]) -> Callable[[], None]:
    """Błąd jednej strony nie wyłącza aplikacji — komunikat + szczegóły w rozwijanym panelu."""

    @functools.wraps(fn)
    def wrapper() -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            log.exception("Błąd strony %s", fn.__name__)
            st.error(f"Nie udało się wyświetlić strony: {e}")
            with st.expander("Szczegóły błędu"):
                st.exception(e)

    return wrapper


def _flash(kind: str, text: str) -> None:
    st.session_state["_flash"] = (kind, text)


def _show_flash() -> None:
    msg = st.session_state.pop("_flash", None)
    if msg:
        kind, text = msg
        st.toast(text, icon="✅" if kind == "success" else "⚠️")


def _go(page: str, **focus: str) -> None:
    for k, v in focus.items():
        st.session_state[k] = v
    st.session_state["_goto"] = page


def _go_material(mid: str) -> None:
    _go("material", focus_material=mid)


def _go_campaign(cid: str) -> None:
    _go("kampanie", focus_campaign=cid)


def _embed_html(page: str, height: int) -> None:
    if hasattr(st, "iframe"):
        st.iframe(page, height=height)
    else:  # starsze wersje Streamlit
        import streamlit.components.v1 as components

        components.html(page, height=height, scrolling=False)


def _selected_rows(event: Any) -> list[int]:
    try:
        sel = event.selection if hasattr(event, "selection") else event["selection"]
        rows = sel.rows if hasattr(sel, "rows") else sel["rows"]
        return [int(r) for r in rows]
    except Exception:  # noqa: BLE001
        return []


def _accepts_kw(fn: Callable[..., Any], name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _offline(cfg: Settings) -> Settings:
    return cfg.model_copy(update={"network": cfg.network.model_copy(update={"enabled": False})})


def _no_llm(cfg: Settings) -> Settings:
    """Przetwarzanie wsadowe bez LLM: lokalny model to ~13 s na materiał, heurystyki — ułamki sekundy."""
    return cfg.model_copy(update={"llm": cfg.llm.model_copy(update={"enabled": False})})


# ---------------------------------------------------------------- wspólne elementy widoków


def _empty_state() -> None:
    st.info(
        "**Katalog roboczy jest pusty** — nie ma jeszcze przetworzonych materiałów.\n\n"
        "Demonstracja na zbiorze syntetycznym (z etykietami kampanii, bez sieci):\n\n"
        "```\npython -m slad demo\n```\n"
        "Własny zbiór (np. przykładowe reklamy NASK w `data/nask/`): `python -m slad run data/nask`.\n\n"
        "Możesz też użyć panelu **„Przetwórz zbiór danych lub wygeneruj demo”** na stronie Przegląd.",
        icon=":material/inbox:",
    )


def _show_load_errors(b: C.Bundle) -> None:
    if b.errors:
        with st.expander(f"⚠️ Problemy z odczytem katalogu roboczego ({len(b.errors)})"):
            for e in b.errors:
                st.text(e)


def _contrib_bar(contrib: Sequence[tuple[str, float, float]], title: str | None = None) -> None:
    """Pasek udziału grup przesłanek w wyniku (≤ 5 segmentów, z etykietami obok)."""
    contrib = [c for c in contrib if c[1] > 0]
    if not contrib:
        return
    segs, labels = [], []
    for g, w, share in contrib:
        color = C.GROUP_COLORS.get(g, C.INK_MUTED)
        name = _html.escape(g)
        segs.append(f'<div style="flex:{share:.4f};background:{color}" title="{name} {share:.0%}"></div>')
        labels.append(
            f'<span><span class="slad-sw" style="background:{color}"></span>{C.GROUP_ICONS.get(g, "•")} '
            f"{name} <b>{share:.0%}</b> ({w:.1f})</span>"
        )
    head = f'<div class="slad-muted">{_html.escape(title)}</div>' if title else ""
    st.html(f'{head}<div class="slad-bar">{"".join(segs)}</div><div class="slad-labels">{"".join(labels)}</div>')


def _evidence_table(evidence: Iterable[Evidence], limit: int | None = None) -> None:
    """Tabela przesłanek Grupa | Przesłanka | Wartość | Waga — HTML, aby tekst się zawijał (bez ukrytych kolumn)."""
    table = C.evidence_html(evidence, limit)
    if table:
        st.html(table)
    else:
        st.caption("Brak przesłanek.")


def _count_table(df: pd.DataFrame, bars: bool = True) -> None:
    """Tabela „wartość → liczba materiałów”; w wąskich kolumnach bez pasków (liczby zawsze widoczne)."""
    if df.empty:
        st.caption("brak danych")
        return
    max_n = max(1, int(df["Materiały"].max()))
    if bars:
        col = st.column_config.ProgressColumn("Materiały", min_value=0, max_value=max_n, format="%d", width=150)
    else:
        col = st.column_config.NumberColumn("Materiały", format="%d", width=80)
    st.dataframe(df, hide_index=True, width="stretch", column_config={"Materiały": col})


def _links_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.caption("Brak powiązań.")
        return
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "Wynik": st.column_config.NumberColumn("Wynik", format="%.1f"),
            "Pokrycie": st.column_config.ProgressColumn("Pokrycie", min_value=0.0, max_value=1.0, format="percent"),
        },
    )


def _material_card(b: C.Bundle, mid: str, key: str) -> None:
    m, fs = b.materials.get(mid), b.features.get(mid)
    _image_or_placeholder(C.first_image(m, fs), 360)
    st.markdown(f"**{C.md_escape(C.shorten(mid, 42))}**")
    if m is not None and m.url:
        st.caption(C.md_escape(C.defang(C.shorten(m.url, 80))))
    snippet = C.text_snippet(m, fs, 100)
    if snippet:
        st.text(snippet)
    st.button("Szczegóły", key=key, on_click=_go_material, args=(mid,), icon=":material/open_in_new:")


def _gallery(b: C.Bundle, ids: Sequence[str], key_prefix: str, per_row: int = 4, page_size: int = 12) -> None:
    ids = list(ids)
    if not ids:
        st.caption("Brak materiałów.")
        return
    n_pages = max(1, math.ceil(len(ids) / page_size))
    page = 1
    if n_pages > 1:
        page = int(
            st.number_input(
                f"Strona galerii (1–{n_pages})", min_value=1, max_value=n_pages, value=1, key=f"{key_prefix}_page"
            )
        )
    chunk = ids[(page - 1) * page_size : page * page_size]
    for start in range(0, len(chunk), per_row):
        cols = st.columns(per_row)
        for col, mid in zip(cols, chunk[start : start + per_row], strict=False):
            with col, st.container(border=True):
                _material_card(b, mid, key=f"{key_prefix}_open_{mid}")


def _render_graph(data: G.GraphData, height: int = 680) -> None:
    if data.legend:
        st.html(G.legend_chips_html(data.legend))
    try:
        page = G.render_html(data, height=height, legend=False)
    except ImportError:
        st.warning("Biblioteka pyvis nie jest zainstalowana — graf jest niedostępny (`pip install pyvis`).")
        return
    except Exception as e:  # noqa: BLE001
        log.exception("Nie udało się zbudować grafu")
        st.error(f"Nie udało się zbudować grafu: {e}")
        return
    _embed_html(page, height + 12)
    st.caption(
        "Najedź na węzeł lub krawędź, aby zobaczyć szczegóły i przesłanki; kliknięcie węzła wyróżnia jego sąsiadów."
    )


def _link_details(b: C.Bundle, lk: Link, mid: str, key: str, expanded: bool) -> None:
    other = C.other_end(lk, mid)
    rel = C.relation_value(lk.relation)
    title = f"{C.RELATION_ICONS.get(rel, '•')} {b.label(other, 70)} — {rel}, wynik {lk.score:.1f}"
    with st.expander(C.md_escape(title), expanded=expanded):
        c1, c2 = st.columns([1, 3], gap="medium")
        with c1:
            _image_or_placeholder(C.first_image(b.materials.get(other), b.features.get(other)), 320)
            st.caption(C.md_escape(b.campaign_label(b.campaign_of.get(other))))
            if other in b.materials:
                st.button(
                    "Otwórz materiał", key=key, on_click=_go_material, args=(other,), icon=":material/open_in_new:"
                )
        with c2:
            st.markdown(C.md_safe(C.link_sentence(lk)))
            _contrib_bar(C.group_contributions(lk.evidence))
            _evidence_table(lk.evidence)


# ---------------------------------------------------------------- panel boczny


def _add_custom_ws() -> None:
    raw = str(st.session_state.get("_ws_new", "")).strip()
    if not raw:
        return
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    custom = list(st.session_state.get("_ws_custom", []))
    if str(p.resolve()) not in custom:
        custom.append(str(p.resolve()))
    st.session_state["_ws_custom"] = custom
    st.session_state["ws_path"] = str(p.resolve())


def _sidebar(cfg: Settings) -> Path:
    default = cfg.path(cfg.workspace)
    extra = [default, *(Path(p) for p in st.session_state.get("_ws_custom", []))]
    options = C.list_workspaces(ROOT / "workspace", extra)
    keys = [str(p) for p in options]
    if st.session_state.get("ws_path") not in keys:
        start = C.preferred_workspace(options, explicit=bool(os.getenv("SLAD_WORKSPACE")))
        st.session_state["ws_path"] = str(start) if start is not None else keys[0]
    with st.sidebar:
        st.selectbox(
            "Katalog roboczy",
            keys,
            key="ws_path",
            format_func=lambda k: C.workspace_label(Path(k), ROOT),
            help="Podkatalogi workspace/ oraz katalog z ustawień (SLAD_WORKSPACE).",
        )
        with st.expander("Inny katalog roboczy"):
            st.text_input("Ścieżka", key="_ws_new", placeholder="workspace/nask")
            st.button("Użyj", key="_ws_add", on_click=_add_custom_ws)
    return Path(st.session_state["ws_path"])


def _sidebar_status(b: C.Bundle, cfg: Settings) -> None:
    with st.sidebar:
        if b.materials:
            st.caption(f"{len(b.materials)} materiałów · {len(b.campaigns)} kampanii · {len(b.links)} powiązań")
        else:
            st.caption("Katalog roboczy jest pusty.")
        if not cfg.network.enabled:
            st.caption("🔌 Tryb offline (bez sieci)")
        if st.button("Odśwież dane", key="_refresh", icon=":material/refresh:"):
            _bundle_cached.clear()
            st.rerun()
        st.caption("Od pojedynczych reklam do kampanii oszustw · HackNation 2026 · NASK „Kolekcjoner Fejków”")


# ---------------------------------------------------------------- 1. Przegląd


def _progress_sink(bar: Any) -> Callable[[float, str], None]:
    """Aktualizacja paska postępu — także z wątków roboczych pipeline'u."""
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

    ctx = get_script_run_ctx()
    lock = threading.Lock()

    def sink(frac: float, text: str) -> None:
        with lock:
            if ctx is not None and get_script_run_ctx() is None:
                add_script_run_ctx(threading.current_thread(), ctx)
            if frac >= 1.0:
                label = "Cechy gotowe — łączenie materiałów i grupowanie w kampanie…"
            else:
                label = f"Ekstrakcja cech: {text}" if text else "Ekstrakcja cech…"
            bar.progress(frac, text=label)

    return sink


class _Throttled:
    """Adapter callbacku postępu: dowolna sygnatura → (ułamek, opis), nie częściej niż co 0,1 s."""

    def __init__(self, sink: Callable[[float, str], None], min_interval: float = 0.1) -> None:
        self.sink = sink
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        frac, text = C.parse_progress(*args, **kwargs)
        now = time.monotonic()
        with self._lock:
            if frac < 1.0 and now - self._last < self.min_interval:
                return
            self._last = now
        try:
            self.sink(frac, text)
        except Exception:  # noqa: BLE001 — pasek postępu nigdy nie przerywa przetwarzania
            log.debug("Aktualizacja postępu nie powiodła się", exc_info=True)


def _run_processing(path: Path, root: Path, cfg: Settings, offline: bool) -> None:
    if not path.exists():
        st.error(f"Nie znaleziono ścieżki: {path}")
        return
    try:
        from slad.pipeline import process_dataset
    except ImportError as e:
        st.error(f"Moduł przetwarzania (slad.pipeline) jest niedostępny: {e}")
        return
    from slad.store import Workspace

    run_cfg = _offline(cfg) if offline else cfg
    bar = st.progress(0.0, text="Przygotowanie…")
    t0 = time.monotonic()
    try:
        summary = process_dataset(path, Workspace(root), run_cfg, progress=_Throttled(_progress_sink(bar)))
    except Exception as e:  # noqa: BLE001
        log.exception("Przetwarzanie zbioru %s nie powiodło się", path)
        st.error(f"Przetwarzanie nie powiodło się: {e}")
        return
    bar.progress(1.0, text="Gotowe")
    msg = f"Przetworzono zbiór „{path.name}” w {time.monotonic() - t0:.0f} s"
    if isinstance(summary, dict) and "materials" in summary:
        msg += (
            f": {summary.get('materials', 0)} materiałów, {summary.get('campaigns', 0)} kampanii, "
            f"{summary.get('links', 0)} powiązań"
        )
    _flash("success", msg + ".")
    st.rerun()


def _processing_panel(cfg: Settings, root: Path, expanded: bool) -> None:
    with st.expander("Przetwórz zbiór danych lub wygeneruj demo", expanded=expanded, icon=":material/play_circle:"):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Przetwórz zbiór danych**")
            st.caption(
                "CSV / JSON / JSONL / XLSX, katalog z obrazami lub plikami HTML, lista adresów URL. "
                "Wyniki trafią do bieżącego katalogu roboczego."
            )
            data_root = cfg.path(cfg.data_root)
            default = data_root / "nask" if (data_root / "nask").exists() else data_root
            path = st.text_input("Ścieżka do zbioru", value=str(default), key="proc_path")
            offline = st.checkbox(
                "Bez sieci (tylko dane wejściowe i pamięć podręczna)", value=not cfg.network.enabled, key="proc_offline"
            )
            use_llm = st.checkbox(
                "LLM do cech tekstowych (wolniej — lokalny model ok. 13 s na materiał)",
                value=False,
                key="proc_llm",
                disabled=not cfg.llm.enabled,
            )
            if st.button("Przetwórz", type="primary", key="proc_run", icon=":material/play_arrow:"):
                _run_processing(Path(path).expanduser(), root, cfg if use_llm else _no_llm(cfg), offline)
        with right:
            st.markdown("**Zbiór syntetyczny (demo)**")
            st.caption(
                "Syntetyczne reklamy z etykietami kampanii — pokaz i ewaluacja bez danych NASK; przetwarzany bez sieci "
                "i bez LLM (kilkanaście sekund)."
            )
            c1, c2, c3 = st.columns(3)
            seed = c1.number_input("Ziarno", min_value=0, max_value=100_000, value=7, key="syn_seed")
            n_campaigns = c2.number_input("Kampanie", min_value=1, max_value=60, value=8, key="syn_nc")
            noise = c3.number_input("Szum", min_value=0, max_value=1000, value=12, key="syn_noise")
            if st.button("Wygeneruj i przetwórz", key="syn_run", icon=":material/science:"):
                try:
                    from slad.ingest.synthetic import generate
                except ImportError as e:
                    st.error(f"Generator zbioru syntetycznego jest niedostępny: {e}")
                    return
                out_dir = cfg.path(cfg.data_root) / "synthetic"
                try:
                    with st.spinner("Generowanie zbioru syntetycznego…"):
                        ds = generate(out_dir, seed=int(seed), n_campaigns=int(n_campaigns), noise=int(noise))
                except Exception as e:  # noqa: BLE001
                    log.exception("Generowanie zbioru syntetycznego")
                    st.error(f"Generowanie nie powiodło się: {e}")
                    return
                _run_processing(Path(ds or out_dir), root, _no_llm(cfg), offline=True)


def _relation_bar(k: dict[str, int]) -> None:
    total = sum(k[r] for r in C.RELATIONS)
    if not total:
        return
    segs, labels = [], []
    for rel in C.RELATIONS:
        n = k[rel]
        if not n:
            continue
        color = C.RELATION_COLORS[rel]
        segs.append(f'<div style="flex:{n / total:.4f};background:{color}" title="{rel}: {n}"></div>')
        labels.append(f'<span><span class="slad-sw" style="background:{color}"></span>{rel} <b>{n}</b></span>')
    st.html(
        '<div class="slad-muted">Powiązania według relacji</div>'
        f'<div class="slad-bar">{"".join(segs)}</div><div class="slad-labels">{"".join(labels)}</div>'
    )


def _run_info(b: C.Bundle) -> None:
    sentence, details, samples = C.run_summary(b.run_info)
    if not sentence:
        return
    with st.expander(C.md_escape(sentence), icon=":material/history:"):
        st.dataframe(details, hide_index=True, width="stretch")
        if samples:
            st.markdown("**Przykładowe błędy i braki danych**")
            for e in samples:
                st.text(e)


@_guard
def page_overview() -> None:
    b, cfg = _b(), _cfg()
    st.title("Przegląd")
    st.caption(f"Katalog roboczy: {C.md_escape(C.workspace_label(b.root, ROOT))}")
    _show_load_errors(b)
    if not b.materials:
        _empty_state()
        _processing_panel(cfg, b.root, expanded=True)
        return

    k = C.overview_counts(b)
    cols = st.columns(5)
    cols[0].metric("Materiały", k["materials"], border=True, help=f"z wyekstrahowanymi cechami: {k['with_features']}")
    cols[1].metric("Kampanie", k["campaigns"], border=True, help="grupy materiałów połączonych wspólnymi śladami")
    cols[2].metric(
        "Powiązania",
        k["links"],
        border=True,
        help=f"duplikaty: {k['duplikat']} · warianty: {k['wariant']} · powiązane: {k['powiązany']}",
    )
    cols[3].metric("Bez powiązań", k["unlinked"], border=True, help="materiały niepołączone z żadnym innym")
    cols[4].metric("Poza kampaniami", k["unassigned"], border=True, help="materiały nieprzypisane do kampanii")
    _relation_bar(k)
    _run_info(b)

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.subheader("Pokrycie cech")
        st.dataframe(
            C.coverage_frame(b),
            hide_index=True,
            width="stretch",
            column_config={
                "Udział": st.column_config.ProgressColumn("Udział", min_value=0.0, max_value=100.0, format="%.0f%%")
            },
        )
        st.caption("Brakujące cechy nie blokują analizy — porównujemy tylko cechy dostępne w obu materiałach.")
    with right:
        st.subheader("Największe kampanie")
        df = C.campaigns_frame(b.campaigns).head(8)
        if df.empty:
            st.caption("Nie wykryto jeszcze kampanii.")
        else:
            st.dataframe(
                df[["ID", "Nazwa", "Materiały", "Spójność"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "ID": st.column_config.TextColumn("ID", width=70),
                    "Nazwa": st.column_config.TextColumn("Nazwa", width=260),
                    "Materiały": st.column_config.ProgressColumn(
                        "Materiały", min_value=0, max_value=int(df["Materiały"].max()), format="%d", width=150
                    ),
                    "Spójność": st.column_config.NumberColumn("Spójność", format="%.1f", width=90),
                },
            )

    st.subheader("Najczęstsze osoby, marki, narracje i domeny")
    feats = list(b.features.values())
    blocks = [
        ("persons", "👤 Osoby"),
        ("brands", "🏷️ Marki i instytucje"),
        ("narratives", "🧭 Narracje"),
        ("domains", "🌐 Domeny"),
    ]
    for row in (blocks[:2], blocks[2:]):
        cols = st.columns(2, gap="large")
        for col, (what, title) in zip(cols, row, strict=True):
            with col:
                st.markdown(f"**{title}**")
                _count_table(C.top_values(feats, what, 8))

    _processing_panel(cfg, b.root, expanded=False)


# ---------------------------------------------------------------- 2. Kampanie


def _campaign_detail(b: C.Bundle, c: Campaign) -> None:
    color, shape = b.styles.get(c.id, (C.UNASSIGNED_COLOR, "dot"))
    glyph = C.SHAPE_GLYPHS.get(shape, "●")
    st.html(
        f'<div class="slad-head"><span class="glyph" style="color:{color}">{glyph}</span>'
        f"{_html.escape(c.id)} · {_html.escape(C.clean_text(c.name))}</div>"
    )
    internal = C.campaign_links(b, c)
    s = c.summary
    cols = st.columns(5)
    cols[0].metric("Materiały", len(c.members), border=True)
    cols[1].metric("Spójność", f"{c.cohesion:.1f}", border=True, help="średni wynik powiązań wewnątrz kampanii")
    cols[2].metric("Powiązania wewnętrzne", len(internal), border=True)
    cols[3].metric("Pierwsze wystąpienie", C.fmt_dt(s.first_seen) or "—", border=True)
    cols[4].metric("Ostatnie wystąpienie", C.fmt_dt(s.last_seen) or "—", border=True)

    st.markdown("#### Podsumowanie")
    st.caption("Liczba materiałów kampanii z daną wartością.")
    indicators = [(C.pretty_indicator(str(v)), n) for v, n in s.indicators]
    st.html(
        C.summary_html(
            [
                ("👤 Osoby", s.persons),
                ("🏷️ Marki i instytucje", s.brands),
                ("🧭 Narracje", s.narratives),
                ("🌐 Domeny", s.domains),
                ("🧩 Wspólne wskaźniki", indicators),
            ]
        )
    )
    if s.keywords:
        st.markdown("**Słowa kluczowe:** " + C.md_escape(", ".join(s.keywords[:25])))

    st.markdown("#### Dlaczego to jedna kampania?")
    _contrib_bar(
        C.group_contributions(e for lk in internal for e in lk.evidence),
        "Udział grup przesłanek we wszystkich powiązaniach wewnątrz kampanii",
    )
    if c.key_evidence:
        st.caption("Kluczowe przesłanki łączące materiały kampanii (najczęstsze silne cechy wspólne):")
        _evidence_table(c.key_evidence)
    agg = C.aggregate_evidence(internal)
    if not agg.empty:
        st.caption("Najczęstsze przesłanki w powiązaniach wewnątrz kampanii (liczba powiązań i łączna waga):")
        st.dataframe(
            C.style_groups(agg),
            hide_index=True,
            width="stretch",
            column_config={"Łączna waga": st.column_config.NumberColumn("Łączna waga", format="%.1f")},
        )
    elif not c.key_evidence:
        st.caption("Brak zapisanych przesłanek dla tej kampanii.")

    t_members, t_graph, t_links = st.tabs(
        [f"🖼️ Materiały ({len(c.members)})", "🕸️ Graf kampanii", f"🔗 Powiązania wewnętrzne ({len(internal)})"]
    )
    with t_members:
        _gallery(b, c.members, key_prefix=f"cg_{c.id}")
    with t_graph:
        a, bcol = st.columns(2)
        hubs = a.toggle("Pokaż wspólne wskaźniki (tracker, domena, IP…)", value=True, key=f"cgh_{c.id}")
        neighbors = bcol.toggle("Pokaż powiązania z materiałami spoza kampanii", value=False, key=f"cgn_{c.id}")
        data = G.build_graph(
            b.materials,
            b.features,
            b.links,
            b.campaigns,
            campaign_id=c.id,
            include_neighbors=neighbors,
            hubs=hubs,
            labels=True,
            styles=b.styles,
        )
        _render_graph(data, height=540)
    with t_links:
        _links_table(C.links_frame(internal, b))


@_guard
def page_campaigns() -> None:
    b = _b()
    st.title("Kampanie")
    if not b.materials:
        _empty_state()
        return
    if not b.campaigns:
        st.info("Nie wykryto kampanii — w bazie nie ma grup powiązanych materiałów.", icon=":material/info:")
        return
    df = C.campaigns_frame(b.campaigns)
    ids = df["ID"].tolist()
    focus = st.session_state.pop("focus_campaign", None)
    if focus in ids:
        st.session_state["campaign_pick"] = focus
    st.caption("Kliknij wiersz tabeli albo wybierz kampanię z listy poniżej.")
    event = st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="campaign_table",
        column_config={
            "Spójność": st.column_config.ProgressColumn(
                "Spójność", min_value=0.0, max_value=max(1.0, float(df["Spójność"].max())), format="%.1f"
            ),
            "Materiały": st.column_config.NumberColumn("Materiały", format="%d"),
        },
    )
    rows = _selected_rows(event)
    if rows and 0 <= rows[0] < len(ids):
        picked = ids[rows[0]]
        if picked != st.session_state.get("_campaign_table_last"):
            st.session_state["_campaign_table_last"] = picked
            st.session_state["campaign_pick"] = picked
    if st.session_state.get("campaign_pick") not in ids:
        st.session_state["campaign_pick"] = ids[0]
    cid = st.selectbox(
        "Kampania",
        ids,
        key="campaign_pick",
        format_func=lambda i: f"{i} · {C.shorten(b.campaign_by_id[i].name, 70)} ({len(b.campaign_by_id[i].members)})",
    )
    st.divider()
    _campaign_detail(b, b.campaign_by_id[cid])


# ---------------------------------------------------------------- 3. Graf powiązań


@_guard
def page_graph() -> None:
    b = _b()
    st.title("Graf powiązań")
    if not b.materials:
        _empty_state()
        return
    st.caption(
        "Węzły to materiały (kolor i kształt = kampania, szary = bez kampanii). Krawędzie to powiązania: "
        "grubość = wynik, odcień i linia = relacja. Najedź kursorem, aby zobaczyć przesłanki."
    )
    top = max(1.0, math.ceil(max((lk.score for lk in b.links), default=0.0)))
    camp_ids = sorted(b.campaign_by_id)
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.2, 1.4, 1.8], gap="medium")
        min_score = c1.slider("Minimalny wynik powiązania", 0.0, float(top), 0.0, 0.5, key="g_min")
        rels = c2.pills("Relacje", list(C.RELATIONS), selection_mode="multi", default=list(C.RELATIONS), key="g_rel")
        camp = c3.selectbox(
            "Kampania",
            ["__all__", *camp_ids],
            key="g_camp",
            format_func=lambda i: "wszystkie kampanie" if i == "__all__" else b.campaign_label(i),
        )
        t1, t2, t3 = st.columns(3, gap="medium")
        hubs = t1.toggle(
            "Podążaj za śladem — wspólne wskaźniki",
            value=False,
            key="g_hubs",
            help="Dodaje węzły wspólnych wskaźników (tracker, domena, IP, favicona, portfel, telefon…), "
            "przez które materiały są połączone.",
        )
        isolated = t2.toggle(
            "Materiały bez powiązań",
            value=len(b.materials) <= 300,
            key="g_iso",
            help="Pokaż także materiały, które nie mają widocznych powiązań.",
        )
        neighbors = t3.toggle(
            "Sąsiedzi spoza kampanii",
            value=False,
            key="g_nb",
            disabled=camp == "__all__",
            help="Przy wybranej kampanii: pokaż też powiązane materiały z innych kampanii.",
        )
        kinds: list[str] = list(G.DEFAULT_HUB_KINDS)
        if hubs:
            kinds = st.multiselect(
                "Rodzaje wskaźników",
                list(G.HUB_KINDS),
                default=list(G.DEFAULT_HUB_KINDS),
                format_func=lambda k: G.HUB_KINDS.get(k, k),
                key="g_kinds",
            )
    campaign_id = None if camp == "__all__" else camp
    data = G.build_graph(
        b.materials,
        b.features,
        b.links,
        b.campaigns,
        min_score=float(min_score),
        relations=list(rels or []),
        campaign_id=campaign_id,
        include_neighbors=bool(neighbors),
        include_isolated=bool(isolated),
        hubs=bool(hubs),
        hub_kinds=kinds or [],
        max_hub_fraction=0.5 if campaign_id is None else None,
        styles=b.styles,
    )
    info = f"Widoczne: {data.n_materials} materiałów · {data.n_links} powiązań"
    if hubs:
        info += f" · {data.n_hubs} wspólnych wskaźników"
    if data.truncated:
        info += " · pokazano tylko najsilniejsze powiązania (limit)"
    st.caption(info)
    _render_graph(data, height=700)

    with st.expander(f"Tabela widocznych powiązań ({len(data.links)})"):
        _links_table(C.links_frame(data.links, b))
    mids = [n["id"] for n in data.nodes if n["id"] in b.materials]
    if mids:
        c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
        pick = c1.selectbox("Szczegóły materiału z grafu", mids, key="g_pick", format_func=lambda i: b.label(i, 90))
        c2.button("Otwórz", key="g_open", on_click=_go_material, args=(pick,), icon=":material/open_in_new:")


# ---------------------------------------------------------------- 4. Materiał


def _material_text(m: Material, fs: FeatureSet | None) -> None:
    if fs is None:
        if m.text:
            st.text(C.shorten(m.text, 4000))
        st.info("Brak wyekstrahowanych cech tekstowych.")
        return
    st.dataframe(C.text_facts(fs), hide_index=True, width="stretch")
    iocs = C.iocs_frame(fs.text.iocs)
    if not iocs.empty:
        st.markdown("**Wskaźniki IOC** (kontakty, portfele, adresy)")
        st.dataframe(iocs, hide_index=True, width="stretch")
    texts = [
        ("Tekst z wejścia", fs.text.input_text or m.text or ""),
        ("Tekst odczytany z obrazów (OCR)", fs.text.ocr_text),
        ("Widoczny tekst strony docelowej", fs.text.page_text),
    ]
    for label, text in texts:
        if text and text.strip():
            with st.expander(f"{label} ({len(text)} znaków)", expanded=label.startswith("Tekst z wejścia")):
                st.text(text[:6000] + ("…" if len(text) > 6000 else ""))


def _material_images(m: Material, fs: FeatureSet | None) -> None:
    images = fs.images if fs else []
    if not images and not m.image_paths:
        st.info("Materiał nie zawiera obrazów.")
        return
    done: set[str] = set()
    for img in images:
        done.add(img.path)
        path = Path(img.path)
        c1, c2 = st.columns([2, 3], gap="medium")
        with c1:
            overlay = None
            if img.faces and _mtime(path) is not None:
                faces_json = json.dumps([f.model_dump(mode="json") for f in img.faces])
                overlay = _faces_cached(str(path), _mtime(path), faces_json)
            if overlay:
                st.image(
                    overlay, width="stretch", caption="Wykryte twarze (zielona ramka = osoba z bazy referencyjnej)"
                )
            else:
                _image_or_placeholder(path, 640)
        with c2:
            st.markdown(f"**{C.md_escape(path.name)}** · {img.width}×{img.height} px")
            hashes = pd.DataFrame(
                [
                    {"Hash": "pHash", "Wartość": img.phash},
                    {"Hash": "dHash", "Wartość": img.dhash},
                    {"Hash": "wHash", "Wartość": img.whash},
                    {"Hash": "colorHash", "Wartość": img.colorhash},
                    {"Hash": "SHA-256", "Wartość": img.sha256},
                ]
            )
            st.dataframe(hashes, hide_index=True, width="stretch")
            colors = [c for c in img.dominant_colors if C.is_hex_color(c)]
            if colors:
                sw = "".join(
                    f'<span><span class="slad-sw" style="background:{c};width:16px;height:16px"></span>{c}</span>'
                    for c in colors[:8]
                )
                st.html(f'<div class="slad-labels"><span>Kolory dominujące:</span>{sw}</div>')
            faces = C.faces_frame([img])
            if not faces.empty:
                st.markdown("**Twarze**")
                st.dataframe(faces, hide_index=True, width="stretch")
    for p in m.image_paths:
        if p not in done:
            st.caption(f"Obraz {C.md_escape(Path(p).name)} — nie udało się wyznaczyć cech (szczegóły w błędach).")
            _image_or_placeholder(Path(p), 480)


def _material_infra(m: Material, fs: FeatureSet | None) -> None:
    if fs is None:
        st.info("Brak wyekstrahowanych cech infrastruktury.")
        return
    inf = fs.infra
    chain = C.redirect_frame(inf)
    if not chain.empty:
        st.markdown("**Łańcuch przekierowań**")
        st.dataframe(chain, hide_index=True, width="stretch")
    table = C.infra_frame(inf, m.observed_at)
    if table.empty:
        st.info(
            "Brak danych infrastruktury (materiał bez adresu URL, tryb offline albo niedostępna strona).",
            icon=":material/cloud_off:",
        )
    else:
        st.dataframe(table, hide_index=True, width="stretch")


@_guard
def page_material() -> None:
    b = _b()
    st.title("Materiał")
    if not b.materials:
        _empty_state()
        return
    ids = list(b.materials)
    focus = st.session_state.pop("focus_material", None)
    if focus in b.materials:
        st.session_state["material_pick"] = focus
    if st.session_state.get("material_pick") not in b.materials:
        st.session_state["material_pick"] = ids[0]
    mid = st.selectbox("Wybierz materiał", ids, key="material_pick", format_func=lambda i: b.label(i, 100))
    m, fs = b.materials[mid], b.features.get(mid)
    cid = b.campaign_of.get(mid)

    head, side = st.columns([3, 2], gap="large")
    with head:
        st.subheader(C.md_escape(mid))
        meta = [("Źródło", C.relation_value(m.source))]
        meta += [("Platforma", m.platform or ""), ("Reklamodawca", m.advertiser or "")]
        meta += [("Zaobserwowano", C.fmt_dt(m.observed_at, with_time=True))]
        meta += [("URL", C.defang(C.shorten(m.url, 160)) if m.url else "")]
        meta += [("Zapisany HTML", Path(m.html_path).name if m.html_path else "")]
        st.markdown("  \n".join(f"**{k}:** {C.md_escape(C.shorten(v, 160))}" for k, v in meta if v))
        if m.url:
            st.caption("Adres w formie „rozbrojonej” (hxxp, [.]) — bez przypadkowego wejścia na stronę oszustów.")
        if m.label:
            st.caption(f"Etykieta referencyjna (tylko do ewaluacji): {C.md_escape(m.label)}")
    with side:
        if cid:
            st.badge(C.md_escape(b.campaign_label(cid)), icon=":material/hub:", color="blue")
            st.button("Otwórz kampanię", key="m_open_campaign", on_click=_go_campaign, args=(cid,))
        else:
            st.badge("bez kampanii", icon=":material/help:", color="gray")
        if fs is not None:
            avail = ", ".join(fs.available) or "brak"
            st.markdown(f"**Dostępne grupy cech:** {C.md_escape(avail)}")
        st.markdown(f"**Powiązania:** {len(b.links_by_material.get(mid, []))}")
    if fs is None:
        st.warning("Brak wyekstrahowanych cech — materiał nie został jeszcze przetworzony.")
    errs = C.all_errors(fs)
    if errs:
        with st.expander(f"⚠️ Braki i błędy danych ({len(errs)})"):
            for e in errs:
                st.text(e)

    t_text, t_img, t_infra, t_raw = st.tabs(["📝 Tekst", "🖼️ Obraz", "🌐 Infrastruktura", "🧾 Dane surowe"])
    with t_text:
        _material_text(m, fs)
    with t_img:
        _material_images(m, fs)
    with t_infra:
        _material_infra(m, fs)
    with t_raw:
        # etykieta kampanii (Material.label) służy wyłącznie ewaluacji — nie pokazujemy jej w danych surowych
        st.json(m.model_dump(mode="json", exclude={"label"}), expanded=False)
        if fs is not None:
            st.json(fs.model_dump(mode="json", exclude={"text": {"minhash"}}), expanded=False)

    st.subheader("Najbardziej podobne materiały")
    links = b.links_by_material.get(mid, [])[:10]
    if not links:
        st.info("Brak powiązań z innymi materiałami w bazie.", icon=":material/link_off:")
        return
    st.dataframe(
        C.nearest_frame(links, mid, b),
        hide_index=True,
        width="stretch",
        column_config={
            "Wynik": st.column_config.NumberColumn("Wynik", format="%.1f"),
            "Pokrycie": st.column_config.ProgressColumn("Pokrycie", min_value=0.0, max_value=1.0, format="percent"),
        },
    )
    for i, lk in enumerate(links):
        _link_details(b, lk, mid, key=f"sim_{mid}_{i}", expanded=i == 0)


# ---------------------------------------------------------------- 5. Analiza nowego materiału


def _save_upload(root: Path, upload: Any, allowed: set[str], default_suffix: str) -> Path:
    data = upload.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:16]
    name = Path(getattr(upload, "name", "") or "plik")
    suffix = name.suffix.lower() if name.suffix.lower() in allowed else default_suffix
    target_dir = root / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{digest}_{C.safe_filename(name.stem, 40)}{suffix}"
    if not target.exists():
        target.write_bytes(data)
    return target.resolve()


def _make_material(url: str | None, image_path: Path | None, html_path: Path | None, text: str | None) -> Material:
    try:
        from slad.ingest import material_from_input
    except ImportError as e:
        log.warning("slad.ingest niedostępny (%s) — materiał budowany lokalnie", e)
        return C.fallback_material(url, image_path, html_path, text)
    return material_from_input(url=url, image_path=image_path, html_path=html_path, text=text)


def _relink(ws: Any, cfg: Settings) -> bool:
    """Przeliczenie powiązań i kampanii po ręcznym dodaniu materiału (gdy pipeline nie obsługuje add=True)."""
    try:
        from slad.link import build
    except ImportError:
        return False
    links, campaigns, stats = build(ws.all_features(), cfg)
    ws.save_links(links)
    ws.save_campaigns(campaigns)
    ws.save_stats(stats)
    return True


def _call_analyze(material: Material, root: Path, cfg: Settings, add: bool) -> tuple[FeatureSet, Assignment]:
    from slad.pipeline import analyze_new
    from slad.store import Workspace

    ws = Workspace(root)
    kwargs: dict[str, Any] = {}
    if _accepts_kw(analyze_new, "ctx"):
        ctx = _extract_ctx(str(root), cfg.model_dump_json())
        if ctx is not None:
            kwargs["ctx"] = ctx
    if _accepts_kw(analyze_new, "add"):
        return analyze_new(material, ws, cfg, add=add, **kwargs)
    fs, assignment = analyze_new(material, ws, cfg, **kwargs)
    if add:
        ws.add_material(material)
        ws.save_features(fs)
        if not _relink(ws, cfg):
            _flash("warning", "Materiał zapisano, ale powiązania trzeba przeliczyć: python -m slad link")
    return fs, assignment


def _run_analysis(root: Path, cfg: Settings, url: str, image: Any, html_up: Any, text: str, offline: bool) -> None:
    url, text = (url or "").strip(), (text or "").strip()
    if not (url or image or html_up or text):
        st.warning("Podaj co najmniej jedno wejście: adres URL, zrzut ekranu, plik HTML albo tekst.")
        return
    run_cfg = _offline(cfg) if offline else cfg
    with st.status("Analiza materiału…", expanded=True) as status:
        try:
            image_path = _save_upload(root, image, {".png", ".jpg", ".jpeg", ".webp"}, ".png") if image else None
            html_path = _save_upload(root, html_up, {".html", ".htm"}, ".html") if html_up else None
            st.write("Przygotowanie materiału…")
            material = _make_material(url or None, image_path, html_path, text or None)
            st.write("Ekstrakcja cech i porównanie z bazą kampanii…")
            fs, assignment = _call_analyze(material, root, run_cfg, add=False)
        except ImportError as e:
            status.update(label="Analiza niedostępna", state="error")
            st.error(f"Moduł analizy (slad.pipeline) jest niedostępny: {e}")
            return
        except Exception as e:  # noqa: BLE001
            log.exception("Analiza nowego materiału nie powiodła się")
            status.update(label="Analiza nie powiodła się", state="error")
            st.error(f"Analiza nie powiodła się: {e}")
            return
        status.update(label="Analiza zakończona", state="complete", expanded=False)
    st.session_state["analysis"] = {
        "root": str(root),
        "material": material,
        "features": fs,
        "assignment": assignment,
        "offline": offline,
        "added": False,
    }


def _show_analysis(b: C.Bundle, cfg: Settings, root: Path, res: dict[str, Any]) -> None:
    material: Material = res["material"]
    fs: FeatureSet = res["features"]
    a: Assignment = res["assignment"]
    st.divider()
    label, color, icon = C.status_view(a)
    top = st.columns([2.4, 1, 1, 1], gap="medium")
    with top[0]:
        st.markdown("#### Wynik analizy")
        st.badge(label, color=color, icon=icon)
        if a.campaign_id:
            st.markdown(f"Kampania: **{C.md_escape(b.campaign_label(a.campaign_id))}**")
            if a.campaign_id in b.campaign_by_id:
                st.button("Otwórz kampanię", key="an_open_campaign", on_click=_go_campaign, args=(a.campaign_id,))
    top[1].metric("Wynik", f"{a.score:.1f}", border=True, help=f"próg przypisania: {cfg.link.assign.min_score}")
    top[2].metric("Najbliższe materiały", len(a.nearest), border=True)
    top[3].metric("Grupy cech", len(fs.available), border=True, help=", ".join(fs.available) or "brak")
    if a.explanation:
        st.info(C.md_safe(a.explanation), icon=":material/lightbulb:")

    if a.nearest:
        st.markdown("#### Najbliższe materiały i przesłanki")
        st.dataframe(
            C.nearest_frame(a.nearest, material.id, b),
            hide_index=True,
            width="stretch",
            column_config={
                "Wynik": st.column_config.NumberColumn("Wynik", format="%.1f"),
                "Pokrycie": st.column_config.ProgressColumn("Pokrycie", min_value=0.0, max_value=1.0, format="percent"),
            },
        )
        for i, lk in enumerate(a.nearest[:5]):
            _link_details(b, lk, material.id, key=f"an_near_{i}", expanded=i == 0)
    else:
        st.caption("W bazie nie znaleziono podobnych materiałów.")

    st.markdown("#### Wyekstrahowane cechy")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        summary = C.features_summary(fs)
        if summary.empty:
            st.caption("Nie udało się wyznaczyć cech.")
        else:
            st.dataframe(summary, hide_index=True, width="stretch")
    with c2:
        img = C.first_image(material, fs)
        if img is not None:
            _image_or_placeholder(img, 480, caption="Analizowany zrzut ekranu")
        errs = C.all_errors(fs)
        if errs:
            with st.expander(f"⚠️ Braki i błędy danych ({len(errs)})"):
                for e in errs:
                    st.text(e)

    st.divider()
    if res.get("added"):
        st.success("Materiał jest w bazie — kampanie i powiązania zostały zaktualizowane.", icon=":material/check:")
        return
    st.caption("Dodanie zapisuje materiał w katalogu roboczym i aktualizuje powiązania oraz kampanie.")
    if st.button("Dodaj do bazy", type="primary", key="an_add", icon=":material/add:"):
        run_cfg = _offline(cfg) if res.get("offline") else cfg
        try:
            with st.spinner("Dodawanie materiału do bazy…"):
                fs2, a2 = _call_analyze(material, root, run_cfg, add=True)
        except Exception as e:  # noqa: BLE001
            log.exception("Dodanie materiału do bazy nie powiodło się")
            st.error(f"Nie udało się dodać materiału: {e}")
            return
        res.update(features=fs2, assignment=a2, added=True)
        st.session_state["analysis"] = res
        _flash("success", f"Dodano materiał {material.id} do bazy.")
        st.rerun()


@_guard
def page_analyze() -> None:
    b, cfg, root = _b(), _cfg(), _root()
    st.title("Analiza nowego materiału")
    st.caption(
        "Podaj adres, zrzut ekranu, zapisany HTML lub tekst reklamy. Materiał zostanie porównany z bazą "
        "znanych kampanii — każde powiązanie z listą przesłanek i wag."
    )
    if not b.materials:
        st.info("Baza jest pusta — analiza pokaże tylko wyekstrahowane cechy. Najpierw przetwórz zbiór (Przegląd).")
    with st.form("analyze_form", border=True):
        url = st.text_input("Adres URL reklamy lub strony docelowej", placeholder="https://…", key="an_url")
        c1, c2 = st.columns(2, gap="medium")
        image = c1.file_uploader("Zrzut ekranu (PNG, JPG, WEBP)", type=["png", "jpg", "jpeg", "webp"], key="an_image")
        html_up = c2.file_uploader("Zapisany kod strony (HTML)", type=["html", "htm"], key="an_html")
        text = st.text_area("Tekst reklamy lub opis", key="an_text", height=110)
        offline = st.checkbox(
            "Bez sieci — nie pobieraj strony, DNS, RDAP ani TLS", value=not cfg.network.enabled, key="an_offline"
        )
        submitted = st.form_submit_button("Analizuj", type="primary", icon=":material/search:", key="an_submit")
    if submitted:
        _run_analysis(root, cfg, url, image, html_up, text, bool(offline))
    res = st.session_state.get("analysis")
    if res and res.get("root") == str(root):
        _show_analysis(b, cfg, root, res)


# ---------------------------------------------------------------- 6. Metodyka


@_guard
def page_methodology() -> None:
    cfg = _cfg()
    st.title("Metodyka")
    doc = ROOT / "docs" / "METODYKA.md"
    text = None
    if doc.is_file():
        try:
            text = doc.read_text(encoding="utf-8")
        except OSError:
            text = None
    body = ""
    if text:
        lines = text.lstrip().splitlines()
        if lines and lines[0].startswith("# "):
            lines = lines[1:]
        body = "\n".join(lines).strip()
    if len(body) >= 200:  # szkic dokumentu (sam nagłówek) → opis wbudowany
        st.markdown(body)
    else:
        st.markdown(C.BUILTIN_METHODOLOGY)
        st.caption("Pełny opis metody: docs/METODYKA.md.")
    st.divider()
    st.subheader("Aktualne wagi przesłanek")
    st.caption(
        "Waga ≈ log-iloraz szans „ta sama kampania”. Wskaźniki częste w zbiorze (np. wspólny CDN) są "
        "osłabiane przez ważenie rzadkością; wynik powiązania to suma wag przesłanek."
    )
    weights = C.weights_frame(cfg.link.weights)
    if weights.empty:
        st.caption("Brak wag w konfiguracji.")
    else:
        st.dataframe(
            C.style_groups(weights),
            hide_index=True,
            width="stretch",
            height=38 + 35 * len(weights),
            column_config={
                "Waga": st.column_config.ProgressColumn(
                    "Waga", min_value=0.0, max_value=max(1.0, float(weights["Waga"].max())), format="%.1f"
                )
            },
        )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Progi relacji")
        st.dataframe(C.thresholds_frame(cfg.link.thresholds), hide_index=True, width="stretch")
    with c2:
        st.subheader("Grupowanie, przypisanie, źródła danych")
        st.dataframe(C.settings_frame(cfg), hide_index=True, width="stretch")


# ---------------------------------------------------------------- 7. Eksport

_EXPORTS: list[tuple[str, str, str, str, str]] = [
    (
        "neo4j",
        "Neo4j (CSV + Cypher)",
        "slad_neo4j.zip",
        "application/zip",
        "Węzły i relacje w CSV + skrypt Cypher do importu.",
    ),
    ("graphml", "GraphML", "slad.graphml", "application/graphml+xml", "Graf dla Gephi, yEd, Cytoscape."),
    ("json", "JSON", "slad.json", "application/json", "Materiały, powiązania z przesłankami i kampanie."),
    ("csv", "Tabele CSV", "slad_tabele.zip", "application/zip", "Kampanie, powiązania i przypisania (Excel)."),
]


_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Tekst od = + - @ w komórkach CSV (Excel) poprzedzamy apostrofem, tak jak w slad/export/neo4j.py (csv_safe)."""
    return df.map(lambda v: "'" + v if isinstance(v, str) and v.startswith(_CSV_FORMULA_PREFIXES) else v)


@st.cache_data(max_entries=16, show_spinner="Przygotowanie eksportu…")
def _export_bytes(kind: str, root: str, sig: tuple) -> tuple[bytes | None, str]:
    try:
        if kind == "csv":
            b = C.load_bundle(root)
            frames = {
                "kampanie.csv": _csv_safe_frame(C.campaigns_frame(b.campaigns)),
                "powiazania.csv": _csv_safe_frame(C.links_frame(b.links)),
                "przypisania.csv": _csv_safe_frame(C.assignments_frame(b)),
            }
            return C.zip_frames(frames), ""
        from slad.store import Workspace

        ws = Workspace(root)
        with tempfile.TemporaryDirectory(prefix="slad-export-") as tmp:
            tmpdir = Path(tmp)
            if kind == "neo4j":
                from slad.export.neo4j import export_neo4j

                out = tmpdir / "neo4j"
                out.mkdir()
                res = export_neo4j(ws, out)
                src = Path(res) if isinstance(res, str | Path) and Path(res).is_dir() else out
                return C.zip_dir(src), ""
            if kind in ("graphml", "json"):
                from slad.export import graph as export_graph

                target = tmpdir / ("slad.graphml" if kind == "graphml" else "slad.json")
                fn = export_graph.export_graphml if kind == "graphml" else export_graph.export_json
                res = fn(ws, target)
                src = target if target.exists() else (Path(res) if isinstance(res, str | Path) else None)
                if src is None or not src.is_file():
                    return None, "Eksport nie utworzył pliku."
                return src.read_bytes(), ""
    except ImportError as e:
        return None, f"Moduł eksportu jest niedostępny ({e})."
    except Exception as e:  # noqa: BLE001
        log.exception("Eksport %s nie powiódł się", kind)
        return None, f"Eksport nie powiódł się: {e}"
    return None, "Nieznany format eksportu."


def _show_report(report: dict[str, Any]) -> None:
    scalars, tables = C.report_tables(report)
    keys = ("f1", "ari", "nmi", "precision", "recall", "accuracy", "trafno")
    highlights = [r for r in scalars.to_dict("records") if any(k in str(r["Metryka"]).lower() for k in keys)][:6]
    if highlights:
        cols = st.columns(len(highlights))
        for col, r in zip(cols, highlights, strict=True):
            col.metric(str(r["Metryka"]), r["Wartość"], border=True)
    if not scalars.empty:
        st.dataframe(scalars, hide_index=True, width="stretch")
    for title, df in tables:
        st.markdown(f"**{C.md_escape(title)}**")
        st.dataframe(df, width="stretch")
    st.download_button(
        "Pobierz raport (JSON)",
        json.dumps(report, ensure_ascii=False, indent=1, default=str),
        file_name="slad_ewaluacja.json",
        mime="application/json",
        key="dl_eval",
    )


@_guard
def page_export() -> None:
    b, cfg, root = _b(), _cfg(), _root()
    st.title("Eksport")
    if not b.materials:
        _empty_state()
        return
    st.caption("Wyniki do dalszej analizy: graf w Neo4j, Gephi lub yEd (GraphML), JSON albo tabele CSV.")
    sig = C.workspace_signature(root)
    cols = st.columns(len(_EXPORTS), gap="medium")
    for col, (kind, title, fname, mime, desc) in zip(cols, _EXPORTS, strict=True):
        # Karty o równej wysokości, przycisk zawsze na dole — niezależnie od długości opisu.
        with col, st.container(border=True, height="stretch"):
            st.markdown(f"**{title}**")
            st.caption(desc)
            st.space("stretch")
            data, err = _export_bytes(kind, str(root), sig)
            if data:
                st.download_button(
                    "Pobierz", data, file_name=fname, mime=mime, key=f"dl_{kind}", icon=":material/download:"
                )
            else:
                st.caption(f"⚠️ {C.md_escape(err)}")
    st.caption(
        "Import do Neo4j: rozpakuj archiwum do katalogu `import/` bazy i uruchom dołączony skrypt Cypher "
        "(np. w Neo4j Browser)."
    )

    st.divider()
    st.subheader("Ewaluacja jakości (gdy znane są etykiety)")
    n_labeled = sum(1 for m in b.materials.values() if m.label)
    if not n_labeled:
        st.info(
            "Materiały nie mają etykiet kampanii — ewaluacja jest dostępna dla zbiorów z etykietami "
            "(np. syntetycznego: `python -m slad demo`).",
            icon=":material/info:",
        )
        return
    st.caption(
        f"Materiały z etykietą: {n_labeled} z {len(b.materials)}. Ewaluacja porównuje wykryte kampanie z etykietami "
        "(precyzja i czułość par, BCubed, ARI/NMI) i sprawdza przypisywanie nowych materiałów (leave-one-out)."
    )
    if st.button("Uruchom ewaluację", type="primary", key="eval_run", icon=":material/fact_check:"):
        try:
            from slad.eval.run import evaluate
            from slad.store import Workspace

            with st.spinner("Ewaluacja…"):
                report = evaluate(Workspace(root), cfg)
            st.session_state["eval_report"] = (sig, report if isinstance(report, dict) else {"wynik": report})
        except ImportError as e:
            st.warning(f"Moduł ewaluacji (slad.eval) jest niedostępny: {e}")
        except Exception as e:  # noqa: BLE001
            log.exception("Ewaluacja nie powiodła się")
            st.error(f"Ewaluacja nie powiodła się: {e}")
    stored = st.session_state.get("eval_report")
    if stored and stored[0] == sig:
        _show_report(stored[1])


# ---------------------------------------------------------------- start


def main() -> None:
    st.set_page_config(page_title="Bloodhound", page_icon=_page_icon(), layout="wide", initial_sidebar_state="expanded")
    st.html(_CSS % {"ink": C.INK, "ink2": C.INK_2, "muted": C.INK_MUTED})
    st.html(f"<style>{C.EVIDENCE_CSS}</style>")
    try:
        st.logo(_logo(), size="large")
    except Exception:  # noqa: BLE001 — logo jest ozdobą
        log.debug("Logo niedostępne", exc_info=True)
    cfg = _load_cfg()
    pages = {
        "przeglad": st.Page(
            page_overview, title="Przegląd", icon=":material/dashboard:", url_path="przeglad", default=True
        ),
        "kampanie": st.Page(page_campaigns, title="Kampanie", icon=":material/hub:", url_path="kampanie"),
        "graf": st.Page(page_graph, title="Graf powiązań", icon=":material/share:", url_path="graf"),
        "material": st.Page(page_material, title="Materiał", icon=":material/description:", url_path="material"),
        "analiza": st.Page(
            page_analyze, title="Analiza nowego materiału", icon=":material/troubleshoot:", url_path="analiza"
        ),
        "metodyka": st.Page(page_methodology, title="Metodyka", icon=":material/menu_book:", url_path="metodyka"),
        "eksport": st.Page(page_export, title="Eksport", icon=":material/download:", url_path="eksport"),
    }
    nav = st.navigation(
        {
            "Analiza": [pages["przeglad"], pages["kampanie"], pages["graf"], pages["material"]],
            "Działania": [pages["analiza"], pages["eksport"]],
            "Informacje": [pages["metodyka"]],
        },
        expanded=True,
    )
    goto = st.session_state.pop("_goto", None)
    if goto in pages:
        st.switch_page(pages[goto])
    root = _sidebar(cfg)
    bundle = _load_bundle(root)
    _STATE.update(cfg=cfg, root=root, bundle=bundle)
    _sidebar_status(bundle, cfg)
    _show_flash()
    nav.run()


main()
