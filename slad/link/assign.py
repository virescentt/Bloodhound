"""Analiza nowego materiału: przypisanie do znanej kampanii, sygnał „nowa kampania” albo brak powiązań.

1. baza = cechy z katalogu roboczego (bez samego materiału), statystyki korpusu (stats.json lub przeliczone),
2. kandydaci: indeks wskaźników, MinHash, pHash, simhash DOM, twarze, TF-IDF (candidates.neighbours),
3. porównanie z każdym kandydatem (evidence.compare) — zostają powiązania ≥ próg „powiązany”,
4. wynik kampanii = najlepszy wynik powiązania z jej materiałem + 0.25 × drugi najlepszy,
5. wynik ≥ link.assign.min_score → PRZYPISANY; inaczej powiązania z materiałami spoza kampanii →
   NOWA KAMPANIA; inaczej BRAK POWIĄZAŃ. Uzasadnienie po polsku wymienia najsilniejsze przesłanki.

Kontekst porównań (indeks TF-IDF, hashe, embeddingi bazy) jest pamiętany dla katalogu roboczego do czasu zmiany
jego plików — analiza kolejnego materiału na bazie kilku tysięcy pozycji trwa ułamek sekundy.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from slad.link.candidates import neighbours
from slad.link.evidence import LinkContext, PairScore, compare_prepared, score_pair, thresholds
from slad.link.stats import corpus_stats
from slad.models import Assignment, AssignmentStatus, Campaign, FeatureSet, Link
from slad.settings import Settings
from slad.store import Workspace

log = logging.getLogger(__name__)

SECOND_WEIGHT = 0.25  # udział drugiego najlepszego powiązania w wyniku kampanii
AMBIGUOUS_RATIO = 0.7  # druga kampania z wynikiem ≥ 70% najlepszej → wzmianka w uzasadnieniu
TOP_REASONS = 3

_CACHE: dict[str, tuple[tuple, LinkContext]] = {}
_LOCK = threading.RLock()


def _mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


def _fingerprint(ws: Workspace) -> tuple:
    feats = ws.root / "features"
    try:
        n_files = sum(1 for _ in feats.iterdir())
    except OSError:
        n_files = -1
    return (_mtime(feats), n_files, _mtime(ws.path("materials.jsonl")), _mtime(ws.path("stats.json")))


def workspace_context(ws: Workspace, cfg: Settings) -> LinkContext:
    """LinkContext bazy katalogu roboczego (pamiętany do zmiany plików cech / materiałów / statystyk)."""
    key = str(Path(ws.root).resolve())
    fp = _fingerprint(ws)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0] == fp:
            hit[1].cfg = cfg
            return hit[1]
    base = ws.all_features()
    stats = ws.stats()
    if stats.n <= 0:
        stats = corpus_stats(base)
    ctx = LinkContext(cfg, base, stats)
    with _LOCK:
        _CACHE[key] = (fp, ctx)
    return ctx


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def _join_pl(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " i " + parts[-1]


def _reasons(links: list[Link], n: int = TOP_REASONS) -> str:
    """Najsilniejsze różne przesłanki z podanych powiązań (bez dopisków o rzadkości)."""
    seen: set[tuple[str, str | None]] = set()
    out: list[str] = []
    for e in sorted((e for lk in links for e in lk.evidence), key=lambda e: (-e.weight, e.kind, e.value or "")):
        key = (e.kind, e.value)
        if key in seen:
            continue
        seen.add(key)
        out.append(e.detail.split(" — ")[0])
        if len(out) >= n:
            break
    return _join_pl(out)


def _fmt(x: float) -> str:
    return f"{x:.1f}"


def _campaign_label(c: Campaign | None, cid: str) -> str:
    if c is None or not c.name or c.name == f"Kampania {cid}":
        return f"kampanii {cid}"
    return f"kampanii {cid} „{c.name}”"


def assign(fs: FeatureSet, ws: Workspace, cfg: Settings) -> Assignment:
    """Przypisanie nowego materiału do kampanii z katalogu roboczego (z uzasadnieniem po polsku)."""
    mid = fs.material_id
    try:
        ctx = workspace_context(ws, cfg)
    except Exception as e:  # noqa: BLE001
        log.exception("Nie udało się wczytać bazy katalogu roboczego")
        return Assignment(material_id=mid, explanation=f"Nie udało się wczytać bazy materiałów: {e}")
    with _LOCK:
        p = ctx.add(fs)
        try:
            cand = sorted(neighbours(p, ctx) - {mid})
        except Exception as e:  # noqa: BLE001
            log.warning("Kandydaci dla %s: %s — porównuję z całą bazą", mid, e)
            cand = sorted(set(ctx.ids) - {mid})
        links: list[Link] = []
        best_weak: PairScore | None = None
        for other_id in cand:
            other = ctx.prep.get(other_id)
            if other is None:
                continue
            try:
                lk = compare_prepared(p, other, ctx)
            except Exception as e:  # noqa: BLE001
                log.warning("Porównanie %s–%s: %s", mid, other_id, e)
                continue
            if lk is not None:
                links.append(lk)
        if not links:  # najbliższy materiał poniżej progu — do wyjaśnienia „dlaczego nie”
            for other_id in cand:
                other = ctx.prep.get(other_id)
                if other is None:
                    continue
                try:
                    ps = score_pair(p.fs, other.fs, ctx)
                except Exception:  # noqa: BLE001
                    continue
                if ps.score > 0 and (best_weak is None or ps.score > best_weak.score):
                    best_weak = ps
    links.sort(key=lambda lk: (-lk.score, lk.b))
    n_base = len(ctx.ids) - (1 if mid in ctx.by_id else 0)
    return _decide(mid, links, ws.campaigns(), cfg, n_base=n_base, n_cand=len(cand), best_weak=best_weak)


def _decide(
    mid: str,
    links: list[Link],
    campaigns: list[Campaign],
    cfg: Settings,
    n_base: int,
    n_cand: int,
    best_weak: PairScore | None = None,
) -> Assignment:
    related, _, _ = thresholds(cfg)
    top_k = max(1, int(cfg.link.assign.top_k or 10))
    min_score = float(cfg.link.assign.min_score)
    by_id = {c.id: c for c in campaigns}
    member_of: dict[str, str] = {}
    for c in campaigns:
        for m in c.members:
            member_of.setdefault(m, c.id)

    per_campaign: dict[str, list[Link]] = {}
    for lk in links:
        cid = member_of.get(lk.b)
        if cid is not None:
            per_campaign.setdefault(cid, []).append(lk)
    scores: dict[str, float] = {}
    for cid, lks in per_campaign.items():
        s = sorted((lk.score for lk in lks), reverse=True)
        scores[cid] = s[0] + (SECOND_WEIGHT * s[1] if len(s) > 1 else 0.0)
    ranked = sorted(scores, key=lambda c: (-scores[c], c))
    nearest = links[:top_k]

    if ranked and scores[ranked[0]] >= min_score:
        cid = ranked[0]
        lks = per_campaign[cid]
        camp = by_id.get(cid)
        size = len(camp.members) if camp else len(lks)
        text = (
            f"Przypisano do {_campaign_label(camp, cid)} (wynik {_fmt(scores[cid])}). "
            f"Najsilniejsze przesłanki: {_reasons(lks)}. "
            f"Powiązano z {len(lks)} z {size} materiałów kampanii (najbliższy: {lks[0].b}, wynik {_fmt(lks[0].score)})."
        )
        if len(ranked) > 1 and scores[ranked[1]] >= AMBIGUOUS_RATIO * scores[cid]:
            other = ranked[1]
            text += f" Uwaga: podobny wynik dla {_campaign_label(by_id.get(other), other)} ({_fmt(scores[other])})."
        return Assignment(
            material_id=mid,
            campaign_id=cid,
            score=round(scores[cid], 3),
            status=AssignmentStatus.ASSIGNED,
            nearest=nearest,
            explanation=text,
        )

    loose = [lk for lk in links if lk.b not in member_of]
    if loose:
        text = (
            f"Materiał nie pasuje do żadnej znanej kampanii, ale jest powiązany z {len(loose)} "
            f"{'materiałem' if len(loose) == 1 else 'materiałami'} spoza kampanii (najbliższy: {loose[0].b}, "
            f"wynik {_fmt(loose[0].score)}) — możliwa nowa kampania. Przesłanki: {_reasons(loose)}."
        )
        if ranked:
            text += (
                f" Najbliższa znana kampania: {ranked[0]} (wynik {_fmt(scores[ranked[0]])} < próg przypisania "
                f"{_fmt(min_score)})."
            )
        return Assignment(
            material_id=mid,
            campaign_id=None,
            score=round(loose[0].score, 3),
            status=AssignmentStatus.NEW_CAMPAIGN,
            nearest=nearest,
            explanation=text,
        )

    if ranked:  # powiązania tylko z kampaniami, ale poniżej progu przypisania
        cid = ranked[0]
        text = (
            f"Nie przypisano: najlepsze dopasowanie to {_campaign_label(by_id.get(cid), cid)} "
            f"(wynik {_fmt(scores[cid])}), poniżej progu przypisania {_fmt(min_score)}. "
            f"Przesłanki: {_reasons(per_campaign[cid])}."
        )
        return Assignment(
            material_id=mid,
            campaign_id=None,
            score=round(scores[cid], 3),
            status=AssignmentStatus.UNLINKED,
            nearest=nearest,
            explanation=text,
        )

    if n_base == 0:
        text = "Baza materiałów jest pusta — brak punktu odniesienia. Materiał może zapoczątkować nową kampanię."
    else:
        text = (
            f"Brak powiązań z {n_base} materiałami bazy (porównano z {n_cand} kandydatami): żaden wynik nie "
            f"osiągnął progu powiązania {_fmt(related)}."
        )
        if best_weak is not None and best_weak.evidence:
            reasons = _join_pl([e.detail.split(" — ")[0] for e in best_weak.evidence[:TOP_REASONS]])
            text += (
                f" Najbliższy materiał: {best_weak.b} (wynik {_fmt(best_weak.score)}) — {reasons}; "
                f"to za mało, by uznać materiały za powiązane."
            )
    return Assignment(
        material_id=mid,
        campaign_id=None,
        score=0.0,
        status=AssignmentStatus.UNLINKED,
        nearest=[],
        explanation=text,
    )
