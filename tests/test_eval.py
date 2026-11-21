"""Metryki grupowania i raport ewaluacji — małe, ręcznie policzone przykłady."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.eval import ari_nmi, bcubed, evaluate, format_report, gold_labels, pairwise_prf  # noqa: E402
from slad.models import Campaign, Evidence, Link, Material, Relation, SourceKind  # noqa: E402
from slad.store import Workspace  # noqa: E402

GOLD = {"a": 1, "b": 1, "c": 1, "d": 2, "e": 2}
PRED = {"a": "X", "b": "X", "c": "Y", "d": "Y", "e": None}


def test_pairwise_hand_computed() -> None:
    # pary przewidziane: (a,b) trafna, (c,d) błędna; wzorcowe: ab, ac, bc, de
    r = pairwise_prf(PRED, GOLD)
    assert r["precision"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(0.25)
    assert r["f1"] == pytest.approx(1 / 3)


def test_bcubed_hand_computed() -> None:
    # precyzja per materiał: 1, 1, 1/2, 1/2, 1 → 0,8; pełność: 2/3, 2/3, 1/3, 1/2, 1/2 → 8/15
    r = bcubed(PRED, GOLD)
    assert r["precision"] == pytest.approx(0.8)
    assert r["recall"] == pytest.approx(8 / 15)
    assert r["f1"] == pytest.approx(2 * 0.8 * (8 / 15) / (0.8 + 8 / 15))


def test_perfect_singletons_and_one_big_cluster() -> None:
    perfect = {k: f"K{v}" for k, v in GOLD.items()}
    for fn in (pairwise_prf, bcubed):
        assert fn(perfect, GOLD) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert ari_nmi(perfect, GOLD) == pytest.approx({"ari": 1.0, "nmi": 1.0})

    singletons: dict[str, str | None] = dict.fromkeys(GOLD)
    assert pairwise_prf(singletons, GOLD) == {"precision": 1.0, "recall": 0.0, "f1": 0.0}
    assert bcubed(singletons, GOLD)["precision"] == pytest.approx(1.0)
    assert pairwise_prf({}, GOLD) == pairwise_prf(singletons, GOLD)  # brak wpisu = singleton

    one = dict.fromkeys(GOLD, "K")
    r = pairwise_prf(one, GOLD)
    assert r["recall"] == 1.0 and r["precision"] == pytest.approx(4 / 10)
    assert bcubed(one, GOLD)["recall"] == pytest.approx(1.0)


def test_unlabeled_ignored_and_label_names_irrelevant() -> None:
    gold = {**GOLD, "u": None}
    pred = {**PRED, "u": "X", "extra": "X"}
    assert pairwise_prf(pred, gold) == pairwise_prf(PRED, GOLD)
    assert bcubed(pred, gold) == bcubed(PRED, GOLD)
    renamed = {k: (None if v is None else f"inna-{v}") for k, v in PRED.items()}
    assert ari_nmi(renamed, GOLD) == pytest.approx(ari_nmi(PRED, GOLD))
    assert ari_nmi(PRED, GOLD)["ari"] < 0.5


def test_empty_inputs() -> None:
    assert pairwise_prf({}, {}) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert bcubed({}, {}) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert ari_nmi({}, {}) == {"ari": 0.0, "nmi": 0.0}


def test_gold_labels_noise_is_unique() -> None:
    ms = [
        Material(id="1", source=SourceKind.TEXT, label="K1"),
        Material(id="2", source=SourceKind.TEXT, label="-1"),
        Material(id="3", source=SourceKind.TEXT, label="noise"),
        Material(id="4", source=SourceKind.TEXT),
    ]
    g = gold_labels(ms)
    assert set(g) == {"1", "2", "3"} and g["1"] == "K1" and g["2"] != g["3"]


def _link(a: str, b: str, score: float, relation: Relation = Relation.RELATED, kind: str = "tracker_id") -> Link:
    ev = Evidence(kind=kind, group="infrastruktura", detail="wspólny identyfikator", weight=score)
    return Link(a=a, b=b, score=score, relation=relation, evidence=[ev])


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path / "ws")
    labels = {"a": "L1", "b": "L1", "c": "L1", "d": "L2", "e": "L2", "f": "L3", "g": "-1", "h": None}
    ws.save_materials(Material(id=k, source=SourceKind.TEXT, text=k, label=v) for k, v in labels.items())
    ws.save_campaigns(
        [
            Campaign(id="K001", name="Kampania A", members=["a", "b", "c"]),
            Campaign(id="K002", name="Kampania B", members=["d", "f"]),
        ]
    )
    ws.save_links(
        [
            _link("a", "b", 8.0, Relation.VARIANT),
            _link("b", "c", 5.0),
            _link("d", "e", 3.0, kind="nameserver"),  # poprawne, ale poniżej progu przypisania
            _link("d", "f", 6.0, kind="asn"),  # błędne: różne etykiety
            _link("c", "h", 9.0, Relation.DUPLICATE),  # materiał bez etykiety — pomijany w trafności
        ]
    )
    return ws


def test_evaluate_report(workspace: Workspace) -> None:
    cfg = SimpleNamespace(link=SimpleNamespace(assign=SimpleNamespace(min_score=4.0)))
    rep = evaluate(workspace, cfg)
    json.dumps(rep)  # raport musi dać się zapisać jako JSON

    c = rep["counts"]
    assert (c["materials"], c["labeled"], c["labels"], c["unique_labels"]) == (8, 7, 4, 2)
    assert (c["campaigns"], c["clustered"], c["links"]) == (2, 5, 5)

    lk = rep["links"]
    assert (lk["evaluated"], lk["correct"]) == (4, 3) and lk["precision"] == pytest.approx(0.75)
    assert lk["by_relation"]["wariant"] == {"n": 1, "correct": 1, "precision": 1.0}
    assert lk["by_relation"]["powiązany"]["correct"] == 2
    assert lk["by_evidence"]["asn"]["precision"] == 0.0

    # najbliższy sąsiad: a→b ✓, b→a ✓, c→h (9, bez etykiety) ✗, d→f (inna etykieta) ✗, e→d (3 < 4) ✗
    asg = rep["assignment"]
    assert (asg["eligible"], asg["correct"], asg["below_threshold"], asg["wrong_count"]) == (5, 2, 1, 2)
    # do kampanii (bez samego materiału): a→K001 ✓, b→K001 ✓, c→K001 ✓, d→K002 (zostaje tylko f: L3) ✗, e→K002 (3 < 4) ✗
    asc = rep["assignment_campaign"]
    assert (asc["correct"], asc["unassigned"]) == (3, 1)
    assert asc["accuracy"] == pytest.approx(0.6)

    # nie łączyć: f (L3) ma powiązanie 6 ≥ 4 → błąd; g (szum) bez powiązań → poprawnie
    snl = rep["should_not_link"]
    assert (snl["eligible"], snl["correct"], snl["in_campaign"]) == (2, 1, 1)
    assert snl["violations"][0]["id"] == "f"

    assert rep["clustering"]["pairwise"]["precision"] == pytest.approx(3 / 4)
    merged = rep["errors"]["merged_campaigns"]
    assert [r["id"] for r in merged] == ["K002"]
    assert rep["errors"]["false_links"][0]["a"] == "d"

    text = format_report(rep)
    for phrase in ("EWALUACJA", "Grupowanie w kampanie", "Powiązania", "Przypisanie nowego materiału", "Nie łączyć"):
        assert phrase in text
    assert "asn" in text  # przesłanka o niskiej trafności


def test_evaluate_empty_workspace(tmp_path: Path) -> None:
    rep = evaluate(Workspace(tmp_path / "pusty"), None)
    assert rep["counts"]["materials"] == 0 and rep["assignment"]["accuracy"] is None
    assert "EWALUACJA" in format_report(rep)
    assert "EWALUACJA" in format_report({})
