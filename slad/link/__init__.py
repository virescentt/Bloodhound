"""Powiązania i kampanie: FeatureSet[] → Link[] (z przesłankami) → Campaign[].

    links, campaigns, stats = build(features, cfg)

Przebieg: statystyki korpusu (rzadkość wartości) → LinkContext (indeks tekstu, hashe, embeddingi) →
pary kandydatów → porównanie par (przesłanki z wagami) → grupowanie (graf + Louvain) → opisy kampanii.
Analiza nowego materiału: `from slad.link.assign import assign` → assign(fs, ws, cfg). Opis metody: docs/METODYKA.md.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime

from slad.link.candidates import candidate_pairs, neighbours
from slad.link.cluster import cluster
from slad.link.evidence import LinkContext, PairScore, compare, compare_prepared, score_pair
from slad.link.indicators import indicators
from slad.link.stats import corpus_stats, rarity_factor
from slad.link.summarize import summarize
from slad.models import Campaign, CorpusStats, FeatureSet, Link
from slad.settings import Settings

log = logging.getLogger(__name__)


def link_context(features: Iterable[FeatureSet], cfg: Settings, stats: CorpusStats | None = None) -> LinkContext:
    """Kontekst porównań dla zbioru materiałów (statystyki liczone, gdy nie podano)."""
    return LinkContext(cfg, features, stats)


def build(
    features: Iterable[FeatureSet],
    cfg: Settings,
    dates: Mapping[str, datetime | None] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[list[Link], list[Campaign], CorpusStats]:
    """Pełne wiązanie zbioru: powiązania (≥ próg „powiązany”), kampanie (K001…, od największej) i statystyki.

    `dates` (opcjonalnie: id → data obserwacji) wypełnia first_seen / last_seen kampanii.
    `progress` (opcjonalnie) — wywoływane jako progress(etap, zrobione, wszystkie).
    """
    by_id: dict[str, FeatureSet] = {}
    for fs in features:
        if fs is not None:
            by_id[fs.material_id] = fs
    feats = list(by_id.values())
    ctx = LinkContext(cfg, feats)  # statystyki korpusu liczone z przygotowanych wskaźników
    stats = ctx.stats
    pairs = sorted(candidate_pairs(feats, ctx))
    log.info("Powiązania: %d materiałów, %d par kandydatów", len(feats), len(pairs))
    ctx.prepare_pairs(pairs)
    links: list[Link] = []
    total = len(pairs)
    step = max(1, total // 50)
    for k, (a, b) in enumerate(pairs):
        try:
            lk = compare_prepared(ctx.prep[a], ctx.prep[b], ctx)
        except Exception as e:  # noqa: BLE001 — jedna para nie zatrzymuje całego zbioru
            log.warning("Porównanie %s–%s nie powiodło się: %s", a, b, e)
            continue
        if lk is not None:
            links.append(lk)
        if progress is not None and (k % step == 0 or k == total - 1):
            try:
                progress("powiązania", k + 1, total)
            except Exception:  # noqa: BLE001
                pass
    clusters = cluster(ctx.ids, links, cfg.link)
    where = {m: k for k, members in enumerate(clusters) for m in members}
    internal: dict[int, list[Link]] = {}
    for lk in links:  # powiązania wewnętrzne każdej kampanii (jeden przebieg zamiast przeglądania wszystkich)
        k = where.get(lk.a)
        if k is not None and where.get(lk.b) == k:
            internal.setdefault(k, []).append(lk)
    campaigns = []
    for i, members in enumerate(clusters, start=1):
        try:
            campaigns.append(
                summarize(
                    f"K{i:03d}",
                    members,
                    ctx.by_id,
                    internal.get(i - 1, []),
                    dates,
                    indicators_by_id={m: ctx.prep[m].ind for m in members if m in ctx.prep},
                )
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Opis kampanii K%03d nie powiódł się: %s", i, e)
            campaigns.append(Campaign(id=f"K{i:03d}", name=f"Kampania K{i:03d}", members=list(members)))
    log.info("Powiązania: %d, kampanie: %d", len(links), len(campaigns))
    return links, campaigns, stats


__all__ = [
    "LinkContext",
    "PairScore",
    "build",
    "candidate_pairs",
    "cluster",
    "compare",
    "corpus_stats",
    "indicators",
    "link_context",
    "neighbours",
    "rarity_factor",
    "score_pair",
    "summarize",
]
