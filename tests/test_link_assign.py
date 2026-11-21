"""Analiza nowego materiału (slad.link.assign) na tymczasowym katalogu roboczym."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from datasketch import MinHash

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.link import build  # noqa: E402
from slad.link.assign import assign, clear_cache, workspace_context  # noqa: E402
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

_PERMS = MinHash(num_perm=128, seed=1, scheme="affine32").permutations


def minhash(text: str) -> list[int]:
    """Sygnatura jak w slad.features.text (szyngle 3 słów, datasketch affine32, seed 1)."""
    words = text.lower().split()
    if len(words) < 3:
        return []
    m = MinHash(num_perm=128, seed=1, scheme="affine32", permutations=_PERMS)
    m.update_batch({" ".join(words[i : i + 3]).encode() for i in range(len(words) - 2)})
    return [int(v) for v in m.hashvalues]


def make(mid: str, text: str = "", *, persons=(), iocs=None, **infra) -> FeatureSet:
    tf = TextFeatures(input_text=text, text=text, persons=list(persons), iocs=iocs or {}, minhash=minhash(text))
    fs = FeatureSet(material_id=mid, text=tf, infra=InfraFeatures(**infra))
    fs.available = ["tekst"] if text else []
    if fs.infra.domains or fs.infra.tracker_ids:
        fs.available.append("infrastruktura")
    return fs


TEXT_A = (
    "Marek Zawadzki ujawnia: platforma Zyskomat Pro zarabia za ciebie {n} zł dziennie, "
    "zarejestruj się dziś, a konsultant oddzwoni w ciągu godziny i pomoże zacząć inwestować"
)
TEXT_B = (
    "Pilne: program rządowy wypłaci każdemu emerytowi {n} zł dopłaty, wypełnij wniosek online "
    "i podaj dane karty, środki trafią na konto jeszcze w tym miesiącu bez zbędnych formalności"
)
PIXEL = "fbpixel:998877665544"


def base_features() -> list[FeatureSet]:
    feats = [
        make(
            f"a{i}",
            TEXT_A.format(n=5000 + 100 * i),
            persons=["Marek Zawadzki"],
            tracker_ids=[PIXEL],
            domains=[f"zysk{i % 2}.test"],
            final_url=f"https://zysk{i % 2}.test/",
        )
        for i in range(4)
    ]
    feats += [
        make(
            f"b{i}",
            TEXT_B.format(n=3000 + 50 * i),
            persons=["Marek Zawadzki"],
            domains=["doplaty-gov.test"],
            final_url="https://doplaty-gov.test/wniosek",
        )
        for i in range(3)
    ]
    # materiał spoza kampanii: unikalny telefon kontaktowy
    feats.append(
        make(
            "solo",
            "Zadzwoń do doradcy pod numer 512 777 888 i odbierz bonus powitalny",
            iocs={"phone": ["512 777 888"]},
        )
    )
    # tło: niepowiązane materiały (realistyczne statystyki rzadkości)
    topics = ["pogoda", "ogród", "kuchnia", "rower", "góry", "morze", "kino", "muzyka", "szachy", "teatr"]
    feats += [make(f"n{i}", f"{t} " * 4 + f"opis numer {i * 7919} bez związku z resztą") for i, t in enumerate(topics)]
    return feats


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    clear_cache()
    cfg = Settings()
    feats = base_features()
    w = Workspace(tmp_path / "ws")
    w.save_materials([Material(id=fs.material_id, source=SourceKind.TEXT, text=fs.text.text) for fs in feats])
    for fs in feats:
        w.save_features(fs)
    links, campaigns, stats = build(feats, cfg)
    w.save_links(links)
    w.save_campaigns(campaigns)
    w.save_stats(stats)
    return w


def campaign_of(w: Workspace, prefix: str) -> str:
    for c in w.campaigns():
        if all(m.startswith(prefix) for m in c.members):
            return c.id
    raise AssertionError(f"brak kampanii {prefix}")


def test_base_has_two_campaigns_and_loose_material(ws: Workspace):
    members = sorted(sorted(c.members) for c in ws.campaigns())
    assert members == [["a0", "a1", "a2", "a3"], ["b0", "b1", "b2"]]
    assert ws.campaign_of("solo") is None


def test_new_material_joins_the_right_campaign(ws: Workspace):
    new = make("new1", "Zupełnie nowy tekst reklamy o kryptowalutach i szybkim zarobku", tracker_ids=[PIXEL])
    res = assign(new, ws, Settings())
    assert res.status == AssignmentStatus.ASSIGNED
    assert res.campaign_id == campaign_of(ws, "a")
    assert res.score >= Settings().link.assign.min_score
    assert res.nearest and all(lk.a == "new1" for lk in res.nearest)
    assert res.nearest == sorted(res.nearest, key=lambda lk: -lk.score)
    assert "Przypisano do kampanii" in res.explanation and "Facebook Pixel 998877665544" in res.explanation


def test_text_variant_without_infrastructure_is_assigned(ws: Workspace):
    new = make("new2", TEXT_A.format(n=6100), persons=["Marek Zawadzki"])
    res = assign(new, ws, Settings())
    assert res.status == AssignmentStatus.ASSIGNED
    assert res.campaign_id == campaign_of(ws, "a")


def test_unrelated_material_is_unlinked(ws: Workspace):
    new = make(
        "new3",
        "Sprzedam rower górski w dobrym stanie, rama aluminiowa, odbiór osobisty w Bydgoszczy",
        persons=["Marek Zawadzki"],  # ta sama osoba to za mało
    )
    res = assign(new, ws, Settings())
    assert res.status == AssignmentStatus.UNLINKED
    assert res.campaign_id is None and res.nearest == []
    assert res.explanation.startswith("Brak powiązań")


def test_link_to_loose_material_signals_new_campaign(ws: Workspace):
    new = make("new4", "Bonus czeka! Napisz lub zadzwoń: +48 512 777 888", iocs={"phone": ["+48 512 777 888"]})
    res = assign(new, ws, Settings())
    assert res.status == AssignmentStatus.NEW_CAMPAIGN
    assert res.nearest[0].b == "solo"
    assert "nowa kampania" in res.explanation and "512777888" in res.explanation


def test_reanalysis_of_known_material_excludes_itself(ws: Workspace):
    known = ws.features("a1")
    res = assign(known, ws, Settings())
    assert res.status == AssignmentStatus.ASSIGNED
    assert res.campaign_id == campaign_of(ws, "a")
    assert all(lk.b != "a1" for lk in res.nearest)


def test_context_is_cached_until_workspace_changes(ws: Workspace):
    cfg = Settings()
    ctx1 = workspace_context(ws, cfg)
    assert workspace_context(ws, cfg) is ctx1
    extra = make("added", TEXT_B.format(n=3999), domains=["doplaty-gov.test"])
    ws.add_material(Material(id="added", source=SourceKind.TEXT, text=extra.text.text))
    ws.save_features(extra)
    ctx2 = workspace_context(ws, cfg)
    assert ctx2 is not ctx1 and "added" in ctx2.by_id


def test_assign_on_empty_workspace(tmp_path: Path):
    clear_cache()
    res = assign(make("x", "jakiś tekst reklamy o zarobkach w internecie"), Workspace(tmp_path / "empty"), Settings())
    assert res.status == AssignmentStatus.UNLINKED
    assert "pusta" in res.explanation


def test_assign_is_fast_with_cached_context(ws: Workspace):
    cfg = Settings()
    assign(make("warm", "rozgrzewka"), ws, cfg)
    t0 = time.perf_counter()
    for i in range(5):
        assign(make(f"q{i}", TEXT_A.format(n=7000 + i), tracker_ids=[PIXEL]), ws, cfg)
    assert (time.perf_counter() - t0) / 5 < 0.5
