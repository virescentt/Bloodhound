"""Pełność dla dużych kampanii: zbiór > 400 materiałów, kampania większa niż limit częstości indeksu (20% zbioru).

100 materiałów z tym samym numerem telefonu + 330 niepowiązanych (unikalne telefony; 120 z nich ma wspólny adres IP —
przesłanka średnia, której duży kubełek nadal jest pomijany). Kampania ma powstać, nowy materiał z tym numerem
ma trafić do niej, a niepowiązane materiały mają zostać bez powiązań.
"""

from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.link import build, link_context, neighbours  # noqa: E402
from slad.link import candidates as cand  # noqa: E402
from slad.link.assign import assign, clear_cache  # noqa: E402
from slad.models import (  # noqa: E402
    AssignmentStatus,
    FeatureSet,
    InfraFeatures,
    Material,
    SourceKind,
    TextFeatures,
)
from slad.settings import Settings  # noqa: E402
from slad.store import Workspace  # noqa: E402

PHONE = "+48 600 100 200"
N_CAMPAIGN = 100
N_OTHER = 330
N_SHARED_IP = 120
SHARED_IP = "198.51.100.7"  # TEST-NET-2 (dopuszczony w indicators._ip_ok)
STAR_LIMIT = 30  # STRONG_BUCKET_MAX w teście gwiazdy (kubełek 100 > 30)


def _text(rng: random.Random, n: int = 25) -> str:
    return " ".join("".join(rng.choice("abcdefghijklmnoprstuwyz") for _ in range(rng.randint(3, 9))) for _ in range(n))


def make(mid: str, text: str, phone: str, ips: tuple[str, ...] = ()) -> FeatureSet:
    tf = TextFeatures(input_text=text, text=text, iocs={"phone": [phone]})
    fs = FeatureSet(material_id=mid, text=tf, infra=InfraFeatures(ips=list(ips)))
    fs.available = ["tekst"]
    return fs


@pytest.fixture(scope="module")
def corpus() -> dict:
    rng = random.Random(11)
    camp = [make(f"c{i:03d}", _text(rng), PHONE) for i in range(N_CAMPAIGN)]
    phones = rng.sample(range(500_000_000, 900_000_000), N_OTHER)
    other = [
        make(f"u{i:03d}", _text(rng), f"+48 {phones[i]}", (SHARED_IP,) if i < N_SHARED_IP else ())
        for i in range(N_OTHER)
    ]
    feats = camp + other
    rng.shuffle(feats)  # środek gwiazdy = pierwszy materiał kampanii w kolejności bazy, nie przypadkiem c000
    links, campaigns, stats = build(feats, Settings())
    return {
        "feats": feats,
        "camp_ids": {fs.material_id for fs in camp},
        "other_ids": {fs.material_id for fs in other},
        "links": links,
        "campaigns": campaigns,
        "stats": stats,
    }


@pytest.fixture(scope="module")
def ws(corpus: dict, tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    clear_cache()
    w = Workspace(tmp_path_factory.mktemp("recall") / "ws")
    feats = corpus["feats"]
    w.save_materials([Material(id=fs.material_id, source=SourceKind.TEXT, text=fs.text.text) for fs in feats])
    for fs in feats:
        w.save_features(fs)
    w.save_links(corpus["links"])
    w.save_campaigns(corpus["campaigns"])
    w.save_stats(corpus["stats"])
    return w


def test_premise_bucket_is_above_frequency_cap(corpus: dict):
    ctx = link_context(corpus["feats"], Settings())
    bi = cand.base_index(ctx)
    assert len(bi.ids) > cand.ALL_PAIRS_MAX  # tryb kandydatów, nie wszystkie pary
    assert bi.df_cap < N_CAMPAIGN <= cand.STRONG_BUCKET_MAX
    assert bi.df_cap < N_SHARED_IP
    assert all("phone" in ctx.prep[m].ind for m in corpus["camp_ids"] | corpus["other_ids"])
    assert all("ip" in ctx.prep[f"u{i:03d}"].ind for i in range(N_SHARED_IP))


def test_strong_bucket_kept_and_weak_bucket_skipped(corpus: dict):
    ctx = link_context(corpus["feats"], Settings())
    pairs = cand._index_pairs(cand.base_index(ctx))
    # telefon (przesłanka silna) — wszystkie pary kampanii; wspólne IP (średnia, 120 > df_cap) — pominięte
    assert pairs == {(a, b) for a, b in combinations(sorted(corpus["camp_ids"]), 2)}


def test_large_operator_campaign_forms_and_unrelated_stay_unlinked(corpus: dict):
    campaigns = corpus["campaigns"]
    assert campaigns, "brak kampanii"
    largest = campaigns[0]
    assert len(largest.members) >= 90
    assert set(largest.members) <= corpus["camp_ids"]
    other = corpus["other_ids"]
    assert not any(lk.a in other or lk.b in other for lk in corpus["links"])
    assert not any(set(c.members) & other for c in campaigns)


def test_neighbours_of_new_material_include_campaign(corpus: dict):
    ctx = link_context(corpus["feats"], Settings())
    p = ctx.add(make("nowy", _text(random.Random(5)), PHONE))
    assert corpus["camp_ids"] <= neighbours(p, ctx)


def test_assign_finds_large_campaign(corpus: dict, ws: Workspace):
    cfg = Settings()
    target = next(c.id for c in ws.campaigns() if "c000" in c.members)
    res = assign(make("nowy1", _text(random.Random(6)), PHONE), ws, cfg)
    assert res.status == AssignmentStatus.ASSIGNED
    assert res.campaign_id == target
    assert res.score >= cfg.link.assign.min_score
    size = len(next(c for c in ws.campaigns() if c.id == target).members)
    assert f"Powiązano z {size} z {size} materiałów kampanii" in res.explanation


def test_assign_unrelated_material_stays_unlinked(ws: Workspace):
    res = assign(make("nowy2", _text(random.Random(7)), "+48 911 222 444"), ws, Settings())
    assert res.status == AssignmentStatus.UNLINKED
    assert res.campaign_id is None


def test_star_above_bucket_limit_is_linear_and_still_forms_campaign(corpus: dict, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cand, "STRONG_BUCKET_MAX", STAR_LIMIT)
    feats = corpus["feats"]
    ctx = link_context(feats, Settings())
    bi = cand.base_index(ctx)
    order = [fs.material_id for fs in feats if fs.material_id in corpus["camp_ids"]]
    hub = order[0]
    pairs = cand._index_pairs(bi)
    assert len(pairs) == N_CAMPAIGN - 1  # gwiazda: k − 1 par zamiast k(k − 1)/2
    assert all(hub in pair for pair in pairs)

    p = ctx.add(make("nowy3", _text(random.Random(8)), PHONE))
    nb = neighbours(p, ctx)
    assert set(order[:STAR_LIMIT]) <= nb  # pierwsze materiały kubełka, w tym środek gwiazdy
    assert len(nb & corpus["camp_ids"]) <= STAR_LIMIT + cand.TFIDF_TOPK  # ograniczona praca dla nowego materiału

    _, campaigns, _ = build(feats, Settings())
    assert campaigns and len(campaigns[0].members) >= 90
    assert set(campaigns[0].members) <= corpus["camp_ids"]
