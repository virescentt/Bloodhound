"""Potok przetwarzania: zbiór → materiały → cechy → powiązania → kampanie (zapis w workspace).

    process_dataset(path, ws, cfg)   pełne przetworzenie zbioru; przyrostowo — zapisane cechy
                                     niezmienionych materiałów są używane ponownie (szybki restart)
    rebuild_links(ws, cfg)           powiązania i kampanie od nowa z zapisanych cech (np. po zmianie wag)
    analyze_new(material, ws, cfg)   nowy materiał → przypisanie do kampanii (add=True: dopisanie do bazy)

Moduły ingest/ i link/ są importowane leniwie; pipeline korzysta wyłącznie z ich kontraktu
(docs/ARCHITEKTURA.md). Błąd jednego materiału nigdy nie przerywa przetwarzania — trafia do
FeatureSet.errors i do podsumowania.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from slad.features import ExtractContext, extract
from slad.models import Assignment, Campaign, FeatureSet, Link, Material
from slad.settings import Settings
from slad.store import Workspace

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]
"""progress(done, total, material_id) — zawsze wywoływany w wątku wołającym (bezpieczne dla Streamlit).
Na starcie ekstrakcji pada wywołanie (0, total, "")."""
StageFn = Callable[[str, int, int], None]
"""stage_progress(etap, zrobione, wszystkie) — postęp łączenia (gdy slad.link.build go raportuje)."""

# Pliki liczone przez inne moduły z całego korpusu — po przebudowie powiązań są nieaktualne.
DERIVED_FILES = ("text_index.pkl",)
RUN_FILE = "run.json"  # podsumowanie ostatniego przetwarzania (UI: przegląd)
MAX_ERROR_SAMPLES = 20
_CAMPAIGN_ID = re.compile(r"^K(\d+)$")


# ------------------------------------------------------------------ przetwarzanie zbioru


def process_dataset(
    path: str | Path,
    ws: Workspace,
    cfg: Settings,
    progress: ProgressFn | None = None,
    limit: int | None = None,
    *,
    force: bool = False,
    ctx: ExtractContext | None = None,
    stage_progress: StageFn | None = None,
) -> dict[str, Any]:
    """Zbiór → workspace: wczytanie, cechy (równolegle, przyrostowo), powiązania, kampanie, statystyki.

    Zwraca podsumowanie: liczby materiałów / cech / powiązań / kampanii, czasy i liczbę materiałów
    z błędami. Błąd łączenia nie gubi policzonych cech — trafia do summary["link_error"]. Zbiór, którego
    nie da się wczytać (ValueError: zły format, za duży plik), nie rzuca wyjątku: summary["load_errors"]
    zawiera komunikat, materiałów jest 0, a workspace pozostaje bez zmian.
    """
    from slad.ingest import load_dataset

    t0 = time.perf_counter()
    path = Path(path)
    load_errors: list[str] = []
    try:
        loaded = load_dataset(path)
    except ValueError as e:  # FileNotFoundError zostaje — brak ścieżki to błąd wywołania
        log.warning("Nie wczytano zbioru %s: %s", path, e)
        loaded = []
        load_errors.append(str(e))
    materials, duplicate_ids = _unique_materials(loaded)
    if limit is not None:
        materials = materials[: max(0, limit)]
    summary: dict[str, Any] = {
        "dataset": str(path),
        "workspace": str(ws.root),
        "materials": len(materials),
        "duplicate_ids": duplicate_ids,
        "load_errors": load_errors,
    }
    if not materials:
        log.warning("Zbiór %s nie zawiera materiałów — workspace bez zmian", path)
        summary.update(_feature_summary([], 0))
        summary.update(_link_summary([], [], 0.0))
        if load_errors:
            summary["error_samples"] = list(load_errors)  # CLI pokazuje próbki błędów; „errors” zostaje liczbą
        summary["time_s"] = round(time.perf_counter() - t0, 2)
        return summary

    previous = {m.id: m for m in _saved_materials(ws)}
    ws.save_materials(materials)
    features, extracted = extract_features(materials, ws, cfg, progress, force=force, previous=previous, ctx=ctx)
    summary["extract_s"] = round(time.perf_counter() - t0, 2)
    summary.update(_feature_summary(features, extracted))
    try:
        summary.update(rebuild_links(ws, cfg, features=features, materials=materials, stage_progress=stage_progress))
    except Exception as e:  # noqa: BLE001 — cechy są już zapisane; `python -m slad link` ponawia łączenie
        log.exception("Łączenie materiałów nie powiodło się")
        summary.update(_link_summary([], [], 0.0))
        summary["link_error"] = f"{type(e).__name__}: {e}"
    summary["time_s"] = round(time.perf_counter() - t0, 2)
    _write_run_info(ws, cfg, summary)
    return summary


def extract_features(
    materials: Sequence[Material],
    ws: Workspace,
    cfg: Settings,
    progress: ProgressFn | None = None,
    *,
    force: bool = False,
    previous: dict[str, Material] | None = None,
    ctx: ExtractContext | None = None,
) -> tuple[list[FeatureSet], int]:
    """Cechy wszystkich materiałów — `cfg.workers` wątków, każdy wynik zapisywany od razu po policzeniu.

    Zapisane cechy są używane ponownie (chyba że force=True). Gdy podano `previous` (materiały
    z poprzedniego uruchomienia), ponownie używane są tylko cechy materiałów niezmienionych.
    Zwraca (cechy w kolejności materiałów, liczba policzonych teraz).
    """
    total = len(materials)
    done = 0
    by_id: dict[str, FeatureSet] = {}
    todo: list[Material] = []
    prev_fp = None if previous is None else {mid: _fingerprint(m) for mid, m in previous.items()}
    _notify(progress, 0, total, "")
    for m in materials:
        fs = None if force else _reusable_features(ws, m, prev_fp)
        if fs is None:
            todo.append(m)
            continue
        by_id[m.id] = fs
        done += 1
        _notify(progress, done, total, m.id)

    if todo:
        ctx = ctx or ExtractContext.from_settings(cfg, ws.root)
        workers = max(1, min(int(cfg.workers or 1), len(todo)))
        log.info("Ekstrakcja cech: %d do policzenia, %d z zapisu, wątki: %d", len(todo), done, workers)
        if workers == 1:
            for m in todo:
                fs = _extract_one(m, ctx)
                ws.save_features(fs)
                by_id[m.id] = fs
                done += 1
                _notify(progress, done, total, m.id)
        else:
            _thread_safe_caches(ctx)
            pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="slad-cechy")
            try:
                futures = {pool.submit(_extract_one, m, ctx): m for m in todo}
                for fut in as_completed(futures):
                    m = futures[fut]
                    fs = fut.result()  # _extract_one nie rzuca wyjątków
                    ws.save_features(fs)
                    by_id[m.id] = fs
                    done += 1
                    _notify(progress, done, total, m.id)
            except BaseException:
                # Ctrl+C / błąd zapisu: nie czekamy na kolejkę; policzone cechy już są na dysku
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            pool.shutdown(wait=True)
    return [by_id[m.id] for m in materials if m.id in by_id], len(todo)


# ------------------------------------------------------------------ powiązania i kampanie


def rebuild_links(
    ws: Workspace,
    cfg: Settings,
    *,
    features: Iterable[FeatureSet] | None = None,
    materials: Iterable[Material] | None = None,
    stage_progress: StageFn | None = None,
) -> dict[str, Any]:
    """Powiązania, kampanie i statystyki od nowa (slad.link.build) — zapis do workspace.

    Identyfikatory kampanii pozostają stabilne względem poprzedniego zapisu (stabilize_campaign_ids),
    a podsumowania kampanii dostają daty obserwacji materiałów.
    """
    from slad.link import build

    t0 = time.perf_counter()
    materials = list(ws.materials() if materials is None else materials)
    features = list(ws.all_features() if features is None else features)
    old_campaigns = _saved_campaigns(ws)
    dates = {m.id: m.observed_at for m in materials if m.observed_at is not None}

    # rozszerzenia kontraktu build(features, cfg) — przekazywane tylko, gdy build je przyjmuje
    extra: dict[str, Any] = {}
    if dates and _accepts(build, "dates"):
        extra["dates"] = dates
    if stage_progress is not None and _accepts(build, "progress"):
        extra["progress"] = stage_progress
    links, campaigns, stats = build(features, cfg, **extra)
    links, campaigns = list(links), list(campaigns)
    campaigns = stabilize_campaign_ids(campaigns, old_campaigns, {fs.material_id for fs in features})
    if "dates" in extra:
        for c in campaigns:
            _fill_dates(c, dates)
    else:
        campaigns = attach_dates(campaigns, materials, features, links)

    ws.save_links(links)
    ws.save_campaigns(campaigns)
    ws.save_stats(stats)
    _invalidate_derived(ws)
    summary = _link_summary(links, campaigns, time.perf_counter() - t0)
    _refresh_run_info(ws, {"materials": len(materials), "features": len(features), **summary})
    return summary


def stabilize_campaign_ids(
    new: list[Campaign],
    old: list[Campaign],
    current_ids: Iterable[str] | None = None,
    *,
    min_jaccard: float = 0.25,
) -> list[Campaign]:
    """Stabilne identyfikatory kampanii między przebudowami (K003 pozostaje K003 po dodaniu materiału).

    Nowa kampania dziedziczy id starej, z którą najbardziej pokrywa się składem (Jaccard ≥ min_jaccard,
    zachłannie od najlepszego dopasowania); pozostałe dostają kolejne numery powyżej dotychczasowych.
    Bez zmian, gdy nie ma poprzednich kampanii, id nie mają postaci K<numer> albo zbiór materiałów
    zmienił się w większości (wtedy numeracja od nowa jest czytelniejsza).
    """
    if not new or not old or not all(_CAMPAIGN_ID.match(c.id) for c in new):
        return new
    old = [c for c in old if _CAMPAIGN_ID.match(c.id)]
    old_of = {mid: c.id for c in old for mid in c.members}
    if not old_of:
        return new
    if current_ids is not None:
        current = set(current_ids)
        if sum(1 for mid in old_of if mid in current) < 0.5 * len(old_of):
            return new
    old_size = {c.id: len(set(c.members)) for c in old}

    candidates: list[tuple[float, int, int, str]] = []
    for i, c in enumerate(new):
        members = set(c.members)
        for oid, inter in Counter(old_of[m] for m in members if m in old_of).items():
            jaccard = inter / (len(members) + old_size[oid] - inter)
            if jaccard >= min_jaccard:
                candidates.append((jaccard, inter, i, oid))
    mapping: dict[int, str] = {}
    taken: set[str] = set()
    for _, _, i, oid in sorted(candidates, key=lambda t: (-t[0], -t[1], t[2], t[3])):
        if i not in mapping and oid not in taken:
            mapping[i] = oid
            taken.add(oid)
    if not mapping:
        return new

    width = max(3, len(_CAMPAIGN_ID.match(new[0].id).group(1)))
    next_n = max(int(_CAMPAIGN_ID.match(c.id).group(1)) for c in old) + 1
    out: list[Campaign] = []
    for i, c in enumerate(new):
        cid = mapping.get(i)
        if cid is None:
            cid = f"K{next_n:0{width}d}"
            next_n += 1
        out.append(c if c.id == cid else _renumbered(c, cid))
    return out


def _renumbered(c: Campaign, cid: str) -> Campaign:
    """Nowe id kampanii — także w nazwie zastępczej typu „Kampania K003” (gdy brak osób/narracji)."""
    name = re.sub(rf"\b{re.escape(c.id)}\b", cid, c.name)
    return c.model_copy(update={"id": cid, "name": name})


def attach_dates(
    campaigns: list[Campaign],
    materials: Iterable[Material],
    features: Iterable[FeatureSet],
    links: Iterable[Link],
) -> list[Campaign]:
    """Podsumowania kampanii z datami obserwacji materiałów (summarize(..., dates=...) → first/last_seen)."""
    dates = {m.id: m.observed_at for m in materials if m.observed_at is not None}
    if not campaigns or not dates:
        return campaigns
    try:
        from slad.link.summarize import summarize
    except ImportError as e:
        log.warning("Brak slad.link.summarize (%s) — uzupełniam same daty", e)
        summarize = None

    by_id = {fs.material_id: fs for fs in features}
    campaign_of = {mid: c.id for c in campaigns for mid in c.members}
    incident: dict[str, list[Link]] = defaultdict(list)  # powiązania dotykające kampanii
    for lk in links:
        for cid in {campaign_of.get(lk.a), campaign_of.get(lk.b)} - {None}:
            incident[cid].append(lk)

    out: list[Campaign] = []
    for c in campaigns:
        updated: Any = None
        if summarize is not None:
            try:
                updated = summarize(c.id, c.members, by_id, incident.get(c.id, []), dates=dates)
            except TypeError as e:  # starsza sygnatura bez `dates`
                log.debug("summarize() bez dat (%s) — uzupełniam daty bezpośrednio", e)
            except Exception as e:  # noqa: BLE001
                log.warning("Podsumowanie kampanii %s z datami nie powiodło się: %s", c.id, e)
        if not isinstance(updated, Campaign):
            updated = c
        _fill_dates(updated, dates)
        out.append(updated)
    return out


# ------------------------------------------------------------------ nowy materiał


def analyze_new(
    material: Material,
    ws: Workspace,
    cfg: Settings,
    add: bool = False,
    *,
    ctx: ExtractContext | None = None,
) -> tuple[FeatureSet, Assignment]:
    """Nowy materiał: cechy (cache w workspace) → przypisanie do kampanii z przesłankami (slad.link.assign).

    add=True: materiał i cechy trafiają do bazy, a powiązania i kampanie są przeliczane od nowa
    (dla ≤ 5 tys. materiałów to sekundy); id kampanii pozostają stabilne. Przypisanie opisuje stan
    bazy PRZED dodaniem; gdy po przeliczeniu materiał należy do kampanii, wyjaśnienie to odnotowuje.
    `ctx` — gotowy ExtractContext (UI trzyma jeden, żeby nie ładować modeli przy każdej analizie).
    """
    ctx = ctx or ExtractContext.from_settings(cfg, ws.root)
    fs = _extract_one(material, ctx)
    assignment = _assign(fs, ws, cfg)
    if not add:
        return fs, assignment

    ws.add_material(material)
    ws.save_features(fs)
    try:
        rebuild_links(ws, cfg)
    except Exception as e:  # noqa: BLE001
        log.exception("Przeliczenie kampanii po dodaniu %s nie powiodło się", material.id)
        note = f"Materiał zapisano w bazie, ale przeliczenie kampanii nie powiodło się: {e}"
        return fs, assignment.model_copy(update={"explanation": f"{assignment.explanation} {note}".strip()})
    cid = ws.campaign_of(material.id)
    if cid and cid != assignment.campaign_id:
        update: dict[str, Any] = {
            "explanation": f"{assignment.explanation} Po dodaniu do bazy materiał należy do kampanii {cid}.".strip()
        }
        if assignment.campaign_id is None:
            update["campaign_id"] = cid
        assignment = assignment.model_copy(update=update)
    return fs, assignment


def load_run_info(ws: Workspace) -> dict[str, Any] | None:
    """Podsumowanie ostatniego process_dataset (run.json) albo None."""
    f = ws.path(RUN_FILE)
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    except (OSError, ValueError) as e:
        log.warning("Nieczytelny %s: %s", f, e)
        return None


# ------------------------------------------------------------------ pomocnicze


def _extract_one(material: Material, ctx: ExtractContext) -> FeatureSet:
    try:
        fs = extract(material, ctx)
    except Exception as e:  # noqa: BLE001 — extract() nie powinien rzucać; ostatnia linia obrony
        log.exception("Ekstrakcja cech %s nie powiodła się", material.id)
        fs = FeatureSet(material_id=material.id, errors=[f"ekstrakcja cech: {e}"])
    if fs.material_id != material.id:
        fs = fs.model_copy(update={"material_id": material.id})
    return fs


def _assign(fs: FeatureSet, ws: Workspace, cfg: Settings) -> Assignment:
    try:
        from slad.link.assign import assign

        return assign(fs, ws, cfg)
    except Exception as e:  # noqa: BLE001 — UI/CLI dostają czytelny wynik zamiast wyjątku
        log.exception("Przypisanie materiału %s nie powiodło się", fs.material_id)
        return Assignment(material_id=fs.material_id, explanation=f"Nie udało się porównać materiału z bazą: {e}")


def _notify(progress: ProgressFn | None, done: int, total: int, material_id: str) -> None:
    if progress is not None:
        progress(done, total, material_id)


def _accepts(fn: Callable[..., Any], name: str) -> bool:
    """Czy funkcja przyjmuje argument nazwany `name` (rozszerzenia kontraktu innych modułów)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _unique_materials(items: Iterable[Material]) -> tuple[list[Material], int]:
    seen: set[str] = set()
    out: list[Material] = []
    duplicates = 0
    for m in items:
        if m.id in seen:
            duplicates += 1
            continue
        seen.add(m.id)
        out.append(m)
    if duplicates:
        log.warning("Pominięto %d materiałów z powtórzonym identyfikatorem", duplicates)
    return out, duplicates


def _fingerprint(m: Material) -> str | None:
    """Odcisk treści materiału (bez etykiety — nie jest cechą) do wykrywania zmian między uruchomieniami."""
    try:
        data = m.model_dump(mode="json", exclude={"label"})
        return json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 — nieserializowalne meta → traktujemy jako zmieniony
        return None


def _reusable_features(ws: Workspace, m: Material, prev_fp: dict[str, str | None] | None) -> FeatureSet | None:
    if prev_fp is not None:
        fp = _fingerprint(m)
        if fp is None or prev_fp.get(m.id) != fp:
            return None
    try:
        return ws.features(m.id)
    except Exception as e:  # noqa: BLE001 — uszkodzony plik cech → liczymy od nowa
        log.warning("Nieczytelne zapisane cechy %s (%s) — liczę od nowa", m.id, e)
        return None


def _saved_materials(ws: Workspace) -> list[Material]:
    try:
        return ws.materials()
    except Exception as e:  # noqa: BLE001
        log.warning("Nieczytelny materials.jsonl (%s) — liczę wszystkie cechy od nowa", e)
        return []


def _saved_campaigns(ws: Workspace) -> list[Campaign]:
    try:
        return ws.campaigns()
    except Exception as e:  # noqa: BLE001
        log.warning("Nieczytelny campaigns.json (%s) — numeracja kampanii od nowa", e)
        return []


def _thread_safe_caches(ctx: ExtractContext) -> None:
    """ExtractContext.cache() tworzy JsonCache leniwie — przy wielu wątkach tworzymy je pod blokadą."""
    if getattr(ctx, "_slad_cache_lock", None) is not None:
        return
    lock = threading.Lock()
    original = ctx.cache

    def cache(name: str) -> Any:
        with lock:
            return original(name)

    ctx.cache = cache  # type: ignore[method-assign]
    ctx._slad_cache_lock = lock  # type: ignore[attr-defined]


def _utc_key(d: datetime) -> datetime:
    return d.replace(tzinfo=UTC) if d.tzinfo is None else d


def _fill_dates(c: Campaign, dates: dict[str, datetime]) -> None:
    seen = [dates[m] for m in c.members if m in dates]
    if not seen:
        return
    if c.summary.first_seen is None:
        c.summary.first_seen = min(seen, key=_utc_key)
    if c.summary.last_seen is None:
        c.summary.last_seen = max(seen, key=_utc_key)


def _invalidate_derived(ws: Workspace) -> None:
    for name in DERIVED_FILES:
        try:
            ws.path(name).unlink(missing_ok=True)
        except OSError as e:
            log.warning("Nie usunięto nieaktualnego %s: %s", name, e)


def _feature_summary(features: list[FeatureSet], extracted: int) -> dict[str, Any]:
    with_errors = [fs for fs in features if fs.errors]
    return {
        "features": len(features),
        "extracted": extracted,
        "cached": len(features) - extracted,
        "available": dict(Counter(g for fs in features for g in fs.available)),
        "llm_used": sum(1 for fs in features if fs.text.llm_used),
        "errors": len(with_errors),
        "error_samples": [f"{fs.material_id}: {fs.errors[0]}" for fs in with_errors[:MAX_ERROR_SAMPLES]],
    }


def _link_summary(links: list[Link], campaigns: list[Campaign], seconds: float) -> dict[str, Any]:
    relations = Counter(lk.relation.value for lk in links)
    return {
        "links": len(links),
        "relations": dict(relations),
        "campaigns": len(campaigns),
        "clustered": len({m for c in campaigns for m in c.members}),
        "largest_campaign": max((len(c.members) for c in campaigns), default=0),
        "link_s": round(seconds, 2),
    }


def _write_run_info(ws: Workspace, cfg: Settings, summary: dict[str, Any]) -> None:
    info = {
        **summary,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "settings": {
            "network": cfg.network.enabled,
            "ocr": cfg.ocr.enabled,
            "llm": cfg.llm.enabled,
            "llm_model": cfg.llm.model if cfg.llm.enabled else None,
            "faces": cfg.faces.enabled,
            "clip": cfg.clip.enabled,
            "workers": cfg.workers,
        },
    }
    _save_run_info(ws, info)


def _refresh_run_info(ws: Workspace, counts: dict[str, Any]) -> None:
    """Po samym przeliczeniu powiązań (`link`, analyze_new(add=True)) liczby w run.json są aktualizowane."""
    info = load_run_info(ws)
    if info is None:
        return
    info.update(counts)
    info["relinked_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    _save_run_info(ws, info)


def _save_run_info(ws: Workspace, info: dict[str, Any]) -> None:
    target = ws.path(RUN_FILE)
    tmp = target.with_name(f".{target.name}.tmp")
    try:
        tmp.write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, target)
    except OSError as e:
        log.warning("Nie zapisano %s: %s", target, e)
