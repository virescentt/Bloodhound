"""Grupowanie w kampanie, opisy kampanii, kandydaci par i pełne wiązanie zbioru (build)."""

from __future__ import annotations

import random
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

from datasketch import MinHash

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.link import build, candidate_pairs, link_context  # noqa: E402
from slad.link.cluster import cluster  # noqa: E402
from slad.link.summarize import summarize  # noqa: E402
from slad.models import (  # noqa: E402
    Evidence,
    FeatureSet,
    ImageFeatures,
    InfraFeatures,
    Link,
    Relation,
    TextFeatures,
)
from slad.settings import LinkSettings, Settings  # noqa: E402

# ---------------------------------------------------------------- pomocnicze


_PERMS = MinHash(num_perm=128, seed=1, scheme="affine32").permutations


def minhash(text: str) -> list[int]:
    """Sygnatura jak w slad.features.text (szyngle 3 słów, datasketch affine32, seed 1)."""
    words = text.lower().split()
    if len(words) < 3:
        return []
    m = MinHash(num_perm=128, seed=1, scheme="affine32", permutations=_PERMS)
    m.update_batch({" ".join(words[i : i + 3]).encode() for i in range(len(words) - 2)})
    return [int(v) for v in m.hashvalues]


def make(mid: str, text: str = "", *, persons=(), narratives=(), keywords=(), phash: str | None = None, **infra):
    tf = TextFeatures(
        input_text=text,
        text=text,
        persons=list(persons),
        narratives=list(narratives),
        keywords=list(keywords),
        minhash=minhash(text),
    )
    images = []
    if phash:
        images.append(
            ImageFeatures(
                path=f"/tmp/{mid}.png",
                sha256="sha-" + mid,
                width=10,
                height=10,
                phash=phash,
                dhash=phash,
                whash=phash,
                colorhash="0",
            )
        )
    fs = FeatureSet(material_id=mid, text=tf, images=images, infra=InfraFeatures(**infra))
    fs.available = [g for g, ok in (("tekst", bool(text)), ("obraz", bool(images))) if ok]
    if fs.infra.domains or fs.infra.tracker_ids:
        fs.available.append("infrastruktura")
    return fs


def lk(a: str, b: str, score: float, relation: Relation = Relation.RELATED, evidence=None) -> Link:
    return Link(a=a, b=b, score=score, relation=relation, evidence=evidence or [])


def clique(prefix: str, n: int, score: float) -> tuple[list[str], list[Link]]:
    ids = [f"{prefix}{i}" for i in range(n)]
    rel = Relation.VARIANT if score >= 7 else Relation.RELATED
    return ids, [lk(a, b, score, rel) for a, b in combinations(ids, 2)]


# ---------------------------------------------------------------- cluster


def test_weak_bridge_does_not_merge_two_campaigns():
    a_ids, a_links = clique("a", 5, 9.0)
    b_ids, b_links = clique("b", 5, 9.0)
    bridge = lk("a0", "b0", 4.5)
    clusters = cluster(a_ids + b_ids + ["lonely"], a_links + b_links + [bridge], LinkSettings())
    assert clusters == [sorted(a_ids), sorted(b_ids)]


def test_strong_multi_edge_connection_keeps_one_campaign():
    a_ids, a_links = clique("a", 4, 9.0)
    b_ids, b_links = clique("b", 4, 9.0)
    cross = [lk("a0", "b0", 8.0), lk("a1", "b1", 8.0), lk("a2", "b2", 8.0)]
    clusters = cluster(a_ids + b_ids, a_links + b_links + cross, LinkSettings())
    assert clusters == [sorted(a_ids + b_ids)]


def test_duplicates_are_never_separated_and_small_components_kept():
    a_ids, a_links = clique("a", 5, 9.0)
    b_ids, b_links = clique("b", 5, 9.0)
    dup = lk("a0", "b0", 13.0, Relation.DUPLICATE)
    clusters = cluster(a_ids + b_ids, a_links + b_links + [dup], LinkSettings())
    together = [c for c in clusters if "a0" in c][0]
    assert "b0" in together
    pair = cluster(["x", "y", "z"], [lk("x", "y", 5.0), lk("y", "z", 3.0)], LinkSettings())
    assert pair == [["x", "y"]]  # krawędź poniżej progu „powiązany” pominięta, singletony pominięte


def test_cluster_order_is_deterministic_and_respects_min_size():
    ids = ["c1", "c2", "a1", "a2", "a3", "b1", "b2"]
    links = [lk("a1", "a2", 8), lk("a2", "a3", 8), lk("b1", "b2", 8), lk("c1", "c2", 8)]
    assert cluster(ids, links, LinkSettings()) == [["a1", "a2", "a3"], ["b1", "b2"], ["c1", "c2"]]
    cfg = LinkSettings()
    cfg.cluster.min_size = 3
    assert cluster(ids, links, cfg) == [["a1", "a2", "a3"]]


def test_hdbscan_method_groups_cliques():
    a_ids, a_links = clique("a", 4, 9.0)
    b_ids, b_links = clique("b", 4, 9.0)
    cfg = LinkSettings()
    cfg.cluster.method = "hdbscan"
    clusters = cluster(a_ids + b_ids + ["x"], a_links + b_links, cfg)
    assert sorted(clusters) == sorted([sorted(a_ids), sorted(b_ids)])


# ---------------------------------------------------------------- summarize


def test_summarize_names_campaign_and_aggregates_evidence():
    feats = {
        f"m{i}": make(
            f"m{i}",
            "reklama",
            persons=["Marek Zawadzki"],
            narratives=["kampania_testowa"],  # tag spoza config/narratives.yaml — opis = sam tag
            keywords=["bitcoin", "zysk"],
            domains=["zysk24.test"],
            tracker_ids=["fbpixel:998877665544"],
        )
        for i in range(3)
    }
    ev = Evidence(
        kind="tracker_id",
        group="infrastruktura",
        detail="wspólny identyfikator Facebook Pixel 998877665544",
        value="fbpixel:998877665544",
        weight=6.0,
    )
    img = Evidence(kind="image_phash", group="obraz", detail="ta sama grafika (pHash Δ=2)", weight=4.0)
    links = [lk("m0", "m1", 10.0, evidence=[ev, img]), lk("m1", "m2", 6.0, evidence=[ev]), lk("m2", "x", 9.0)]
    dates = {"m0": datetime(2026, 11, 1), "m2": datetime(2026, 11, 9)}
    camp = summarize("K007", ["m2", "m0", "m1"], feats, links, dates=dates)
    assert camp.id == "K007"
    assert camp.members == ["m0", "m1", "m2"]
    assert camp.name == "Marek Zawadzki · kampania testowa · zysk24.test"
    assert camp.cohesion == 8.0  # powiązanie z materiałem spoza kampanii pominięte
    assert camp.summary.persons == [("Marek Zawadzki", 3)]
    assert ("tracker_id:fbpixel:998877665544", 3) in camp.summary.indicators
    assert camp.summary.first_seen == datetime(2026, 11, 1) and camp.summary.last_seen == datetime(2026, 11, 9)
    top = camp.key_evidence[0]
    assert top.kind == "tracker_id" and top.weight == 12.0 and "w 2 powiązaniach" in top.detail
    assert any(e.kind == "image_phash" and "w 1 powiązaniu" in e.detail for e in camp.key_evidence)


def test_summarize_without_features_has_fallback_name():
    camp = summarize("K002", ["q1", "q2"], {}, [])
    assert camp.name == "Kampania K002" and camp.cohesion == 0.0


# ---------------------------------------------------------------- build (całość)


TEXT_A = (
    "Marek Zawadzki ujawnia: platforma Zyskomat Pro zarabia za ciebie 5000 zł dziennie, "
    "zarejestruj się dziś, a konsultant oddzwoni w ciągu godziny i pomoże zacząć"
)
TEXT_B = (
    "Pilne: program rządowy wypłaci każdemu emerytowi 3000 zł dopłaty, wypełnij wniosek "
    "online i podaj dane karty, środki trafią na konto jeszcze w tym miesiącu"
)


def test_build_links_and_clusters_two_campaigns():
    feats = []
    for i in range(4):
        feats.append(
            make(
                f"a{i}",
                TEXT_A.replace("5000", str(5000 + 100 * i)),
                persons=["Marek Zawadzki"],
                tracker_ids=["fbpixel:998877665544"],
                domains=[f"zysk{i % 2}.test"],
                final_url=f"https://zysk{i % 2}.test/",
                phash="8f3a5c7e9b1d2f40",
            )
        )
    for i in range(3):
        feats.append(
            make(
                f"b{i}",
                TEXT_B.replace("3000", str(3000 + 50 * i)),
                persons=["Marek Zawadzki"],  # ta sama osoba w innej kampanii — nie może scalać
                domains=["doplaty-gov.test"],
                final_url="https://doplaty-gov.test/wniosek",
            )
        )
    feats.append(make("noise", "przepis na sernik z rodzynkami i skórką pomarańczową, pieczony godzinę"))
    links, campaigns, stats = build(feats, Settings())
    assert stats.n == len(feats)
    assert [c.members for c in campaigns] == [["a0", "a1", "a2", "a3"], ["b0", "b1", "b2"]]
    assert [c.id for c in campaigns] == ["K001", "K002"]
    assert all({lk.a[0], lk.b[0]} != {"a", "b"} for lk in links)  # żadnego powiązania między kampaniami
    assert not any("noise" in (lk.a, lk.b) for lk in links)
    for link in links:
        assert link.a < link.b
        assert abs(link.score - sum(e.weight for e in link.evidence)) < 1e-6
        assert link.evidence == sorted(link.evidence, key=lambda e: -e.weight)


def test_build_handles_empty_and_single_inputs():
    assert build([], Settings()) == ([], [], build([], Settings())[2])
    links, campaigns, stats = build([make("only", TEXT_A)], Settings())
    assert links == [] and campaigns == [] and stats.n == 1


# ---------------------------------------------------------------- kandydaci (duży zbiór)


def _random_text(rng: random.Random, n: int = 30) -> str:
    words = ["".join(rng.choice("abcdefghijklmnoprstuwyz") for _ in range(rng.randint(3, 9))) for _ in range(n)]
    return " ".join(words)


def test_candidate_pairs_scale_and_recall():
    rng = random.Random(3)
    feats = []
    expected = set()
    for g in range(60):  # 60 kampanii po 7 materiałów + 30 szumu = 450 (> próg porównania wszystkich par)
        base = _random_text(rng)
        tracker = f"fbpixel:{rng.randrange(10**14, 10**15)}"
        ph = f"{rng.getrandbits(64):016x}"
        members = []
        for k in range(7):
            mid = f"g{g:02d}m{k}"
            members.append(mid)
            if k % 3 == 0:  # wspólny tracker
                feats.append(make(mid, _random_text(rng), tracker_ids=[tracker]))
            elif k % 3 == 1:  # niemal ten sam tekst
                feats.append(make(mid, base + " " + _random_text(rng, 2)))
            else:  # przerobiona grafika (pHash Δ=1)
                flipped = int(ph, 16) ^ (1 << rng.randrange(64))
                feats.append(make(mid, _random_text(rng), phash=f"{flipped:016x}"))
        trackers = [m for k, m in enumerate(members) if k % 3 == 0]
        texts = [m for k, m in enumerate(members) if k % 3 == 1]
        images = [m for k, m in enumerate(members) if k % 3 == 2]
        for group in (trackers, texts, images):
            expected |= {tuple(sorted(p)) for p in combinations(group, 2)}
    feats += [make(f"z{i:02d}", _random_text(rng)) for i in range(30)]
    t0 = time.perf_counter()
    ctx = link_context(feats, Settings())
    pairs = candidate_pairs(feats, ctx)
    elapsed = time.perf_counter() - t0
    n = len(feats)
    assert n > 400
    assert len(pairs) < n * (n - 1) // 2 // 10  # znacznie mniej niż wszystkie pary
    recall = len(expected & pairs) / len(expected)
    assert recall >= 0.95, recall
    assert all(a < b for a, b in pairs)
    assert elapsed < 20
