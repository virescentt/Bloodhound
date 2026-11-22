"""Eksport grafu powiązań (networkx): GraphML (Gephi, yEd, Cytoscape) i JSON node-link (D3, własne narzędzia).

Węzły = materiały (źródło, adres, kampania, osoby, narracje, domeny…), krawędzie = powiązania
(weight = wynik, relacja, przesłanki). Kampania jest atrybutem węzła — w Gephi: kolorowanie wg „campaign”.
GraphML przyjmuje tylko wartości skalarne, więc listy są tam łączone znakiem „|”; JSON zachowuje listy
i pełne przesłanki (kind, group, detail, value, similarity, weight) oraz listę kampanii.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx

from slad.models import FeatureSet, Link, Material
from slad.store import Workspace

log = logging.getLogger(__name__)

SEP = "|"


def build_graph(ws: Workspace, *, flat: bool = True) -> nx.Graph:
    """Graf materiałów i powiązań z workspace. flat=True: atrybuty skalarne (wymóg GraphML)."""
    materials = ws.materials()
    campaigns = ws.campaigns()
    campaign_of = {mid: c for c in campaigns for mid in c.members}
    g = nx.Graph()
    g.graph.update(
        {
            "name": "Bloodhound",
            "workspace": ws.root.name,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "materials": len(materials),
            "campaigns": len(campaigns),
        }
    )
    for m in materials:
        c = campaign_of.get(m.id)
        attrs = _node_attrs(m, _features(ws, m.id), c.id if c else "", c.name if c else "")
        g.add_node(m.id, **(_flatten(attrs) if flat else attrs))
    for lk in ws.links():
        for end in (lk.a, lk.b):
            if end not in g:  # powiązanie z materiałem spoza materials.jsonl — węzeł zastępczy
                g.add_node(end, label=end, type="material")
        attrs = _edge_attrs(lk, flat)
        g.add_edge(lk.a, lk.b, **(_flatten(attrs) if flat else attrs))
    return g


def export_graphml(ws: Workspace, path: str | Path) -> Path:
    """GraphML (Gephi / yEd / Cytoscape): węzły = materiały, krawędzie = powiązania z wagą."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(build_graph(ws, flat=True), path)
    return path


def export_json(ws: Workspace, path: str | Path) -> Path:
    """JSON node-link: {"graph", "nodes", "links", "campaigns"} — pełne przesłanki każdego powiązania."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g = build_graph(ws, flat=False)
    try:
        data = nx.node_link_data(g, edges="links")
    except TypeError:  # starsze networkx: klucz "links" domyślnie
        data = nx.node_link_data(g)
    data["campaigns"] = [c.model_dump(mode="json") for c in ws.campaigns()]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return path


# ------------------------------------------------------------------ atrybuty


def _node_attrs(m: Material, fs: FeatureSet | None, campaign: str, campaign_name: str) -> dict[str, Any]:
    fs = fs or FeatureSet(material_id=m.id)
    return {
        "label": _label(m, fs),
        "type": "material",
        "source": m.source.value,
        "url": m.url or "",
        "final_url": fs.infra.final_url or "",
        "platform": m.platform or "",
        "advertiser": m.advertiser or "",
        "observed_at": m.observed_at.isoformat() if m.observed_at else "",
        "campaign": campaign,
        "campaign_name": campaign_name,
        "persons": list(fs.text.persons),
        "face_identities": sorted({f.identity for img in fs.images for f in img.faces if f.identity}),
        "brands": list(fs.text.brands),
        "narratives": list(fs.text.narratives),
        "domains": list(fs.infra.domains),
        "trackers": list(fs.infra.tracker_ids),
        "available": list(fs.available),
        "images": len(m.image_paths),
        "errors": len(fs.errors),
    }


def _edge_attrs(lk: Link, flat: bool) -> dict[str, Any]:
    evidence = sorted(lk.evidence, key=lambda e: e.weight, reverse=True)
    return {
        "weight": float(lk.score),
        "relation": lk.relation.value,
        "coverage": float(lk.coverage),
        "groups": sorted({e.group for e in evidence}),
        "kinds": sorted({e.kind for e in evidence}),
        "evidence": (
            [f"{e.detail} ({e.weight:+.1f})" for e in evidence]
            if flat
            else [e.model_dump(mode="json") for e in evidence]
        ),
    }


def _label(m: Material, fs: FeatureSet) -> str:
    for candidate in (fs.infra.title, fs.infra.domains[0] if fs.infra.domains else None, m.text, m.url):
        if candidate and candidate.strip():
            text = " ".join(candidate.split())
            return text if len(text) <= 60 else text[:59] + "…"
    return m.id


def _flatten(attrs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, list | tuple | set):
            out[key] = SEP.join(str(v).replace(SEP, "/") for v in value if v not in (None, ""))
        elif isinstance(value, bool | int | float | str):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def _features(ws: Workspace, mid: str) -> FeatureSet | None:
    try:
        return ws.features(mid)
    except Exception as e:  # noqa: BLE001
        log.warning("Nieczytelne cechy %s: %s", mid, e)
        return None
