"""Eksport do Neo4j: pliki CSV (węzły i relacje) + import.cypher (LOAD CSV i przykładowe zapytania).

Model grafu:
    (:Material)-[:LINKED {score, relation, coverage, groups, kinds, evidence}]->(:Material)
    (:Material)-[:MEMBER_OF]->(:Campaign)
    (:Material)-[:HAS_INDICATOR]->(:Indicator {key, kind, value, materials})

Listy w polach CSV są łączone znakiem „|” (w Cypher: split(row.pole, '|')). Pliki w UTF-8 bez BOM.
Etykieta kampanii z danych (Material.label) NIE jest eksportowana — służy wyłącznie ewaluacji.
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from slad.models import Campaign, FeatureSet, Link, Material
from slad.store import Workspace

log = logging.getLogger(__name__)

SEP = "|"
MAX_TEXT = 500  # tyle znaków tekstu materiału trafia do węzła (podgląd w Neo4j Browser)

MATERIAL_COLUMNS = (
    "id", "source", "url", "final_url", "platform", "advertiser", "observed_at", "campaign", "title", "text",
    "persons", "face_identities", "brands", "narratives", "domains", "trackers", "available", "images", "faces",
    "errors",
)  # fmt: skip
CAMPAIGN_COLUMNS = (
    "id", "name", "size", "cohesion", "first_seen", "last_seen", "persons", "brands", "narratives", "domains",
    "indicators", "keywords", "key_evidence",
)  # fmt: skip
INDICATOR_COLUMNS = ("key", "kind", "value", "materials")
LINK_COLUMNS = ("a", "b", "score", "relation", "coverage", "groups", "kinds", "evidence")
MEMBER_COLUMNS = ("material", "campaign")
HAS_INDICATOR_COLUMNS = ("material", "key", "kind", "value")

IMPORT_CYPHER = """\
// Ślad Fejka — import grafu materiałów i kampanii do Neo4j (5.x; 4.4 także).
//
// 1) Pliki *.csv z tego katalogu muszą leżeć w katalogu `import` bazy. Najprościej Neo4j w Dockerze,
//    uruchomiony W TYM KATALOGU (Neo4j Browser: http://localhost:7474, login neo4j / slad-fejka):
//      docker run --rm -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/slad-fejka -v "$PWD":/import neo4j:5
// 2) Wczytanie (w drugim terminalu, w tym samym katalogu):
//      docker exec -i $(docker ps -qf ancestor=neo4j:5) cypher-shell -u neo4j -p slad-fejka < import.cypher
//    albo: cypher-shell -u neo4j -p slad-fejka -f import.cypher   (albo wklej polecenia w Neo4j Browser)
// Import jest idempotentny (MERGE) — można go powtarzać po kolejnym eksporcie.
// Bardzo duże zbiory (>100 tys. wierszy): opakuj LOAD CSV w CALL { ... } IN TRANSACTIONS OF 1000 ROWS
// (w Neo4j Browser z prefiksem :auto).

CREATE CONSTRAINT slad_material_id IF NOT EXISTS FOR (m:Material) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT slad_campaign_id IF NOT EXISTS FOR (c:Campaign) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT slad_indicator_key IF NOT EXISTS FOR (i:Indicator) REQUIRE i.key IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///nodes_materials.csv' AS row
MERGE (m:Material {id: row.id})
SET m.source = row.source, m.url = row.url, m.final_url = row.final_url, m.platform = row.platform,
    m.advertiser = row.advertiser, m.observed_at = row.observed_at, m.campaign = row.campaign,
    m.title = row.title, m.text = row.text,
    m.persons = split(row.persons, '|'), m.face_identities = split(row.face_identities, '|'),
    m.brands = split(row.brands, '|'), m.narratives = split(row.narratives, '|'),
    m.domains = split(row.domains, '|'), m.trackers = split(row.trackers, '|'),
    m.available = split(row.available, '|'),
    m.images = toInteger(row.images), m.faces = toInteger(row.faces), m.errors = toInteger(row.errors);

LOAD CSV WITH HEADERS FROM 'file:///nodes_campaigns.csv' AS row
MERGE (c:Campaign {id: row.id})
SET c.name = row.name, c.size = toInteger(row.size), c.cohesion = toFloat(row.cohesion),
    c.first_seen = row.first_seen, c.last_seen = row.last_seen,
    c.persons = split(row.persons, '|'), c.brands = split(row.brands, '|'),
    c.narratives = split(row.narratives, '|'), c.domains = split(row.domains, '|'),
    c.indicators = split(row.indicators, '|'), c.keywords = split(row.keywords, '|'),
    c.key_evidence = split(row.key_evidence, '|');

LOAD CSV WITH HEADERS FROM 'file:///nodes_indicators.csv' AS row
MERGE (i:Indicator {key: row.key})
SET i.kind = row.kind, i.value = row.value, i.materials = toInteger(row.materials);

LOAD CSV WITH HEADERS FROM 'file:///rels_links.csv' AS row
MATCH (a:Material {id: row.a})
MATCH (b:Material {id: row.b})
MERGE (a)-[l:LINKED]->(b)
SET l.score = toFloat(row.score), l.relation = row.relation, l.coverage = toFloat(row.coverage),
    l.groups = split(row.groups, '|'), l.kinds = split(row.kinds, '|'), l.evidence = split(row.evidence, '|');

LOAD CSV WITH HEADERS FROM 'file:///rels_member_of.csv' AS row
MATCH (m:Material {id: row.material})
MATCH (c:Campaign {id: row.campaign})
MERGE (m)-[:MEMBER_OF]->(c);

LOAD CSV WITH HEADERS FROM 'file:///rels_has_indicator.csv' AS row
MATCH (m:Material {id: row.material})
MATCH (i:Indicator {key: row.key})
MERGE (m)-[:HAS_INDICATOR]->(i);

// ------------------------------------------------------------------ przykładowe zapytania
// Kampanie od największej:
//   MATCH (c:Campaign)<-[:MEMBER_OF]-(m:Material)
//   RETURN c.id, c.name, count(m) AS materialy, c.cohesion ORDER BY materialy DESC;
//
// Dlaczego materiały są powiązane — przesłanki z wagami (podmień ID):
//   MATCH (a:Material {id: 'ID'})-[l:LINKED]-(b:Material)
//   RETURN b.id, b.campaign, l.score, l.relation, l.evidence ORDER BY l.score DESC;
//
// Wspólna infrastruktura RÓŻNYCH kampanii (ten sam operator? kandydaci do scalenia):
//   MATCH (c1:Campaign)<-[:MEMBER_OF]-(:Material)-[:HAS_INDICATOR]->(i:Indicator)
//         <-[:HAS_INDICATOR]-(:Material)-[:MEMBER_OF]->(c2:Campaign)
//   WHERE c1.id < c2.id
//   RETURN c1.id, c2.id, i.kind, i.value, count(*) AS wystapienia ORDER BY wystapienia DESC LIMIT 50;
//
// Najczęściej współdzielone wskaźniki (trackery, portfele krypto, telefony, domeny…):
//   MATCH (i:Indicator) WHERE i.materials > 1
//   RETURN i.kind, i.value, i.materials ORDER BY i.materials DESC LIMIT 25;
//
// Materiały spoza kampanii, ale z powiązaniami (kandydaci na nową kampanię):
//   MATCH (m:Material)-[l:LINKED]-(n:Material)
//   WHERE NOT EXISTS { (m)-[:MEMBER_OF]->(:Campaign) }
//   RETURN m.id, m.url, collect(n.id) AS powiazane, max(l.score) AS najlepszy ORDER BY najlepszy DESC;
//
// Graf jednej kampanii z powiązaniami (wizualizacja w Neo4j Browser):
//   MATCH (c:Campaign {id: 'K001'})<-[:MEMBER_OF]-(m:Material)
//   OPTIONAL MATCH p = (m)-[:LINKED]-(:Material)-[:MEMBER_OF]->(c)
//   RETURN c, m, p;
//
// Osoby, których wizerunek wykorzystuje najwięcej kampanii:
//   MATCH (m:Material)-[:MEMBER_OF]->(c:Campaign) UNWIND m.persons AS osoba
//   RETURN osoba, count(DISTINCT c) AS kampanie, count(m) AS materialy ORDER BY kampanie DESC LIMIT 20;
"""


def export_neo4j(ws: Workspace, out_dir: str | Path, *, min_materials: int = 1) -> list[Path]:
    """Zapisuje CSV węzłów i relacji + import.cypher do `out_dir`; zwraca listę zapisanych plików.

    min_materials > 1 pomija wskaźniki występujące w mniejszej liczbie materiałów (mniejszy graf).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    materials = ws.materials()
    known = {m.id for m in materials}
    features = _load_features(ws, materials)
    links = ws.links()
    campaigns = ws.campaigns()
    campaign_of = {mid: c.id for c in campaigns for mid in c.members}

    link_rows = [_link_row(lk) for lk in links if lk.a in known and lk.b in known]
    if len(link_rows) < len(links):
        log.warning("Pominięto %d powiązań z nieznanymi materiałami", len(links) - len(link_rows))
    indicator_rows, has_rows = _indicator_rows(materials, features, min_materials)

    written = [
        _write_csv(
            out / "nodes_materials.csv",
            MATERIAL_COLUMNS,
            (_material_row(m, features.get(m.id), campaign_of.get(m.id)) for m in materials),
        ),
        _write_csv(out / "nodes_campaigns.csv", CAMPAIGN_COLUMNS, (_campaign_row(c) for c in campaigns)),
        _write_csv(out / "nodes_indicators.csv", INDICATOR_COLUMNS, indicator_rows),
        _write_csv(out / "rels_links.csv", LINK_COLUMNS, link_rows),
        _write_csv(
            out / "rels_member_of.csv",
            MEMBER_COLUMNS,
            ((mid, c.id) for c in campaigns for mid in c.members if mid in known),
        ),
        _write_csv(out / "rels_has_indicator.csv", HAS_INDICATOR_COLUMNS, has_rows),
    ]
    cypher = out / "import.cypher"
    cypher.write_text(IMPORT_CYPHER, encoding="utf-8")
    written.append(cypher)
    log.info(
        "Eksport Neo4j: %d materiałów, %d kampanii, %d wskaźników → %s",
        len(materials),
        len(campaigns),
        len(indicator_rows),
        out,
    )
    return written


# ------------------------------------------------------------------ wiersze


def _material_row(m: Material, fs: FeatureSet | None, campaign: str | None) -> tuple[Any, ...]:
    fs = fs or FeatureSet(material_id=m.id)
    faces = [f for img in fs.images for f in img.faces]
    text = fs.text.text or m.text or ""
    return (
        m.id,
        m.source.value,
        m.url,
        fs.infra.final_url,
        m.platform,
        m.advertiser,
        m.observed_at,
        campaign,
        fs.infra.title,
        _clip(text, MAX_TEXT),
        fs.text.persons,
        sorted({f.identity for f in faces if f.identity}),
        fs.text.brands,
        fs.text.narratives,
        fs.infra.domains,
        fs.infra.tracker_ids,
        fs.available,
        len(m.image_paths),
        len(faces),
        len(fs.errors),
    )


def _campaign_row(c: Campaign) -> tuple[Any, ...]:
    s = c.summary
    return (
        c.id,
        c.name,
        len(c.members),
        round(c.cohesion, 4),
        s.first_seen,
        s.last_seen,
        _values(s.persons),
        _values(s.brands),
        _values(s.narratives),
        _values(s.domains),
        _values(s.indicators),
        s.keywords,
        [e.detail for e in c.key_evidence[:10]],
    )


def _link_row(lk: Link) -> tuple[Any, ...]:
    evidence = sorted(lk.evidence, key=lambda e: e.weight, reverse=True)
    return (
        lk.a,
        lk.b,
        round(lk.score, 4),
        lk.relation.value,
        round(lk.coverage, 4),
        sorted({e.group for e in evidence}),
        sorted({e.kind for e in evidence}),
        [f"{e.detail} ({e.weight:+.1f})" for e in evidence],
    )


def _indicator_rows(
    materials: Sequence[Material], features: dict[str, FeatureSet], min_materials: int
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    try:
        from slad.link.indicators import indicators
    except ImportError as e:
        log.warning("Brak slad.link.indicators (%s) — eksport bez węzłów Indicator", e)
        return [], []
    per_material: dict[str, list[tuple[str, str]]] = {}
    df: Counter[tuple[str, str]] = Counter()
    for m in materials:
        fs = features.get(m.id)
        if fs is None:
            continue
        try:
            found = indicators(fs) or {}
        except Exception as e:  # noqa: BLE001 — jeden materiał nie psuje eksportu
            log.warning("Wskaźniki materiału %s: %s", m.id, e)
            continue
        pairs = sorted({(str(kind), str(v)) for kind, values in found.items() for v in values if v not in (None, "")})
        per_material[m.id] = pairs
        df.update(pairs)
    keep = {pair for pair, n in df.items() if n >= max(1, min_materials)}
    indicator_rows = [(_key(k, v), k, v, n) for (k, v), n in sorted(df.items()) if (k, v) in keep]
    has_rows = [(mid, _key(k, v), k, v) for mid, pairs in per_material.items() for k, v in pairs if (k, v) in keep]
    return indicator_rows, has_rows


# ------------------------------------------------------------------ pomocnicze


def _load_features(ws: Workspace, materials: Iterable[Material]) -> dict[str, FeatureSet]:
    out: dict[str, FeatureSet] = {}
    for m in materials:
        try:
            fs = ws.features(m.id)
        except Exception as e:  # noqa: BLE001
            log.warning("Nieczytelne cechy %s: %s", m.id, e)
            continue
        if fs is not None:
            out[m.id] = fs
    return out


def _key(kind: str, value: str) -> str:
    return f"{kind}:{value}"


def _values(pairs: Iterable[Any]) -> list[str]:
    """[(wartość, liczba), …] z podsumowania kampanii → same wartości (kolejność: od najczęstszych)."""
    return [str(p[0]) if isinstance(p, list | tuple) else str(p) for p in pairs]


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return value
    if isinstance(value, list | tuple | set):
        items = (_cell(v) for v in value)
        return SEP.join(str(v).replace(SEP, "/") for v in items if v not in ("", None))
    return " ".join(str(value).split())  # bez znaków nowej linii — czytelniej w Neo4j i arkuszach


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(v) for v in row])
    return path
