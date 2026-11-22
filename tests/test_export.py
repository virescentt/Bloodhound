"""Testy eksportu (slad.export): Neo4j CSV + Cypher, GraphML, JSON node-link. Bez sieci."""

from __future__ import annotations

import csv
import json
import sys
import types
from datetime import datetime
from pathlib import Path

import networkx as nx
import pytest

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.export import build_graph, export_graphml, export_json, export_neo4j  # noqa: E402
from slad.export.neo4j import _cell, _write_csv  # noqa: E402
from slad.models import (  # noqa: E402
    Campaign,
    CampaignSummary,
    Evidence,
    FeatureSet,
    InfraFeatures,
    Link,
    Material,
    Relation,
    SourceKind,
    TextFeatures,
)
from slad.store import Workspace  # noqa: E402

NEO4J_FILES = {
    "nodes_materials.csv",
    "nodes_campaigns.csv",
    "nodes_indicators.csv",
    "rels_links.csv",
    "rels_member_of.csv",
    "rels_has_indicator.csv",
    "import.cypher",
}


def make_workspace(root) -> Workspace:
    """Mały, spójny workspace: 3 materiały, 2 powiązania, 1 kampania (m1+m2), m3 poza kampanią."""
    ws = Workspace(root)
    ws.save_materials(
        [
            Material(
                id="m1",
                source=SourceKind.URL,
                url="https://zysk-teraz.pl/a",
                platform="facebook",
                observed_at=datetime(2026, 11, 2),
                label="TAJNA_ETYKIETA",
            ),
            Material(
                id="m2", source=SourceKind.TEXT, text='Linia 1\nlinia 2, "cudzysłów" | kreska', label="TAJNA_ETYKIETA"
            ),
            Material(id="m3", source=SourceKind.IMAGE, image_paths=["/tmp/nie-ma.png"]),
        ]
    )
    ws.save_features(
        FeatureSet(
            material_id="m1",
            text=TextFeatures(text="inwestuj 5000 zł dziennie", persons=["Jan Kowalski"], narratives=["inwestycje"]),
            infra=InfraFeatures(domains=["zysk-teraz.pl"], tracker_ids=["ga:UA-1234-1"], title="Zysk teraz!"),
            available=["tekst", "infrastruktura"],
        )
    )
    ws.save_features(
        FeatureSet(
            material_id="m2",
            text=TextFeatures(text='Linia 1\nlinia 2, "cudzysłów" | kreska', persons=["Jan Kowalski"]),
            infra=InfraFeatures(domains=["zysk-teraz.pl"], tracker_ids=["ga:UA-1234-1"]),
            available=["tekst", "infrastruktura"],
        )
    )
    ws.save_features(FeatureSet(material_id="m3", errors=["obraz nie-ma.png: brak pliku"]))
    ev = [
        Evidence(
            kind="tracker_id",
            group="infrastruktura",
            detail="wspólny Google Analytics UA-1234-1",
            value="ga:UA-1234-1",
            weight=6.0,
        ),
        Evidence(kind="person", group="osoby", detail="ta sama osoba: Jan Kowalski", value="Jan Kowalski", weight=2.0),
    ]
    ws.save_links(
        [
            Link(a="m1", b="m2", score=8.0, relation=Relation.VARIANT, evidence=ev, coverage=0.67),
            Link(a="m2", b="m-usuniety", score=4.5, relation=Relation.RELATED, evidence=ev[:1]),
        ]
    )
    ws.save_campaigns(
        [
            Campaign(
                id="K001",
                name="Inwestycje — Jan Kowalski",
                members=["m1", "m2"],
                cohesion=8.0,
                summary=CampaignSummary(
                    persons=[("Jan Kowalski", 2)],
                    domains=[("zysk-teraz.pl", 2)],
                    keywords=["inwestuj", "zysk"],
                    first_seen=datetime(2026, 11, 2),
                ),
                key_evidence=ev,
            )
        ]
    )
    return ws


@pytest.fixture
def fake_indicators(monkeypatch):
    mod = types.ModuleType("slad.link.indicators")

    def indicators(fs):
        if fs.material_id == "m3":
            raise ValueError("zepsuty materiał")  # jeden materiał nie może popsuć eksportu
        return {
            "domain": set(fs.infra.domains),
            "tracker_id": set(fs.infra.tracker_ids),
            "person": set(fs.text.persons),
        }

    mod.indicators = indicators
    monkeypatch.setitem(sys.modules, "slad.link.indicators", mod)


def read_csv(path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_export_neo4j_files_and_content(tmp_path, fake_indicators):
    ws = make_workspace(tmp_path / "ws")
    written = export_neo4j(ws, tmp_path / "neo4j")

    assert {p.name for p in written} == NEO4J_FILES
    out = tmp_path / "neo4j"
    materials = read_csv(out / "nodes_materials.csv")
    assert [r["id"] for r in materials] == ["m1", "m2", "m3"]
    m1, m2, m3 = materials
    assert m1["campaign"] == "K001" and m1["domains"] == "zysk-teraz.pl" and m1["observed_at"].startswith("2026-11-02")
    assert m2["text"] == 'Linia 1 linia 2, "cudzysłów" | kreska'  # bez nowych linii, cudzysłowy zachowane
    assert m3["campaign"] == "" and m3["errors"] == "1"
    raw = (out / "nodes_materials.csv").read_text(encoding="utf-8")
    assert "label" not in raw.splitlines()[0] and "TAJNA_ETYKIETA" not in raw  # etykieta tylko do ewaluacji
    assert not raw.startswith("﻿")  # bez BOM (Neo4j wczytałby go do nazwy kolumny)

    campaigns = read_csv(out / "nodes_campaigns.csv")
    assert campaigns[0]["id"] == "K001" and campaigns[0]["size"] == "2" and campaigns[0]["persons"] == "Jan Kowalski"

    links = read_csv(out / "rels_links.csv")
    assert len(links) == 1  # powiązanie z nieznanym materiałem pominięte
    assert links[0]["relation"] == "wariant" and float(links[0]["score"]) == 8.0
    assert links[0]["kinds"] == "person|tracker_id"
    assert links[0]["evidence"].split("|")[0].startswith("wspólny Google Analytics")

    assert read_csv(out / "rels_member_of.csv") == [
        {"material": "m1", "campaign": "K001"},
        {"material": "m2", "campaign": "K001"},
    ]
    indicators = {r["key"]: int(r["materials"]) for r in read_csv(out / "nodes_indicators.csv")}
    assert indicators == {"domain:zysk-teraz.pl": 2, "person:Jan Kowalski": 2, "tracker_id:ga:UA-1234-1": 2}
    has = read_csv(out / "rels_has_indicator.csv")
    assert {(r["material"], r["kind"]) for r in has} >= {("m1", "tracker_id"), ("m2", "domain")}

    cypher = (out / "import.cypher").read_text(encoding="utf-8")
    for name in NEO4J_FILES - {"import.cypher"}:
        assert f"file:///{name}" in cypher
    assert cypher.count("LOAD CSV WITH HEADERS") == 6 and "CREATE CONSTRAINT" in cypher


def test_export_neo4j_min_materials_filters_rare_indicators(tmp_path, fake_indicators):
    ws = make_workspace(tmp_path / "ws")
    (tmp_path / "ws" / "features" / "m2.json").unlink()
    export_neo4j(ws, tmp_path / "neo4j", min_materials=2)
    assert read_csv(tmp_path / "neo4j" / "nodes_indicators.csv") == []
    assert read_csv(tmp_path / "neo4j" / "rels_has_indicator.csv") == []


def test_export_neo4j_without_indicator_module(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "slad.link.indicators", None)  # import → ImportError
    ws = make_workspace(tmp_path / "ws")
    written = export_neo4j(ws, tmp_path / "neo4j")
    assert {p.name for p in written} == NEO4J_FILES
    assert read_csv(tmp_path / "neo4j" / "nodes_indicators.csv") == []
    assert len(read_csv(tmp_path / "neo4j" / "nodes_materials.csv")) == 3


def test_export_empty_workspace(tmp_path, fake_indicators):
    ws = Workspace(tmp_path / "pusty")
    written = export_neo4j(ws, tmp_path / "neo4j")
    assert {p.name for p in written} == NEO4J_FILES
    header = (tmp_path / "neo4j" / "nodes_materials.csv").read_text(encoding="utf-8").splitlines()
    assert header == ["id,source,url,final_url,platform,advertiser,observed_at,campaign,title,text,persons,"
                      "face_identities,brands,narratives,domains,trackers,available,images,faces,errors"]  # fmt: skip
    g = nx.read_graphml(export_graphml(ws, tmp_path / "g.graphml"))
    assert g.number_of_nodes() == 0
    data = json.loads(export_json(ws, tmp_path / "g.json").read_text(encoding="utf-8"))
    assert data["nodes"] == [] and data["links"] == [] and data["campaigns"] == []


def test_export_graphml_roundtrip(tmp_path):
    ws = make_workspace(tmp_path / "ws")
    path = export_graphml(ws, tmp_path / "out" / "slad.graphml")
    g = nx.read_graphml(path)
    assert set(g.nodes) == {"m1", "m2", "m3", "m-usuniety"}  # węzeł zastępczy dla brakującego materiału
    assert g.nodes["m1"]["campaign"] == "K001" and g.nodes["m1"]["label"] == "Zysk teraz!"
    assert g.nodes["m1"]["persons"] == "Jan Kowalski"
    assert "label" in g.nodes["m3"] and g.nodes["m3"]["campaign"] == ""
    edge = g.edges["m1", "m2"]
    assert edge["weight"] == 8.0 and edge["relation"] == "wariant"
    assert edge["evidence"].startswith("wspólny Google Analytics")


def test_export_json_keeps_full_evidence(tmp_path):
    ws = make_workspace(tmp_path / "ws")
    data = json.loads(export_json(ws, tmp_path / "slad.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    assert nodes["m1"]["persons"] == ["Jan Kowalski"] and nodes["m1"]["domains"] == ["zysk-teraz.pl"]
    link = next(lk for lk in data["links"] if {lk["source"], lk["target"]} == {"m1", "m2"})
    assert link["weight"] == 8.0 and link["evidence"][0]["kind"] == "tracker_id"
    assert link["evidence"][0]["weight"] == 6.0 and link["groups"] == ["infrastruktura", "osoby"]
    assert data["campaigns"][0]["id"] == "K001" and data["graph"]["campaigns"] == 1


def test_build_graph_flat_attributes_are_scalars(tmp_path):
    g = build_graph(make_workspace(tmp_path / "ws"), flat=True)
    for _, attrs in g.nodes(data=True):
        assert all(isinstance(v, str | int | float | bool) for v in attrs.values())
    for _, _, attrs in g.edges(data=True):
        assert all(isinstance(v, str | int | float | bool) for v in attrs.values())


def test_export_with_real_indicator_module(tmp_path):
    pytest.importorskip("slad.link.indicators")
    ws = make_workspace(tmp_path / "ws")
    written = export_neo4j(ws, tmp_path / "neo4j")
    assert {p.name for p in written} == NEO4J_FILES
    kinds = {r["kind"] for r in read_csv(tmp_path / "neo4j" / "nodes_indicators.csv")}
    assert "tracker_id" in kinds or "domain" in kinds


def test_csv_text_starting_with_formula_char_is_escaped(tmp_path):
    assert _cell('=HYPERLINK("x")') == '\'=HYPERLINK("x")'
    assert _cell("@admin") == "'@admin" and _cell("+48 600") == "'+48 600"
    assert _cell(["=cmd", "ok"]) == "'=cmd|ok"  # listy: escapowany jest cały tekst komórki
    assert _cell("zwykły tekst") == "zwykły tekst" and _cell("a=b") == "a=b"
    assert _cell(12.5) == 12.5 and _cell(-3) == -3  # liczby zostają liczbami
    path = _write_csv(tmp_path / "f.csv", ["a", "b"], [['=HYPERLINK("x")', 12.5]])
    assert read_csv(path) == [{"a": '\'=HYPERLINK("x")', "b": "12.5"}]
