"""Graf powiązań dla UI: węzły = materiały (kolor/kształt = kampania, szary = bez kampanii),
krawędzie = powiązania (grubość = wynik, odcień i wzór linii = relacja), opcjonalnie węzły-wskaźniki
(wspólny tracker / domena / IP / favicona …) — widok „podążaj za śladem”.

Czyste funkcje, bez Streamlit. HTML powstaje w pyvis z biblioteką vis-network wbudowaną w stronę
(``cdn_resources="in_line"``) — graf działa bez dostępu do internetu. Podpowiedzi (title) są zwykłym
tekstem (vis-network 9 wstawia je przez ``innerText``), etykiety legendy przechodzą przez ``html.escape``.
"""

from __future__ import annotations

import html as _html
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from slad.models import Campaign, FeatureSet, Link, Material
from slad.ui import components as C

HUB_KINDS: dict[str, str] = {
    "tracker": "identyfikator śledzący",
    "domain": "domena",
    "ip": "adres IP",
    "favicon": "favicona (mmh3)",
    "wallet": "portfel kryptowalut",
    "phone": "telefon",
    "email": "e-mail",
    "telegram": "Telegram",
    "whatsapp": "WhatsApp",
    "tls": "certyfikat TLS",
    "form": "adres formularza",
    "nameserver": "serwer nazw (NS)",
    "face": "twarz (osoba z bazy)",
}
HUB_SHORT: dict[str, str] = {
    "tracker": "tracker",
    "domain": "domena",
    "ip": "IP",
    "favicon": "favicon",
    "wallet": "portfel",
    "phone": "tel.",
    "email": "e-mail",
    "telegram": "Telegram",
    "whatsapp": "WhatsApp",
    "tls": "TLS",
    "form": "formularz",
    "nameserver": "NS",
    "face": "twarz",
}
DEFAULT_HUB_KINDS: tuple[str, ...] = ("tracker", "domain", "ip", "favicon", "wallet", "phone", "telegram")
HUB_COLOR = C.INK_2
HUB_EDGE_COLOR = C.INK_MUTED
HIGHLIGHT_COLOR = C.CATEGORICAL[0]
MAX_TOOLTIP_LINES = 16


@dataclass
class GraphData:
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    legend: list[dict[str, Any]] = field(default_factory=list)
    n_materials: int = 0
    n_links: int = 0
    n_hubs: int = 0
    truncated: bool = False
    links: list[Link] = field(default_factory=list)  # widoczne powiązania (po filtrach i limicie)


def tooltip(lines: Iterable[str], max_lines: int = MAX_TOOLTIP_LINES, width: int = 140) -> str:
    """Tekst podpowiedzi: bez znaków sterujących, przycięty; „href” rozbijamy, aby pyvis nie
    przełączył podpowiedzi w tryb HTML (innerHTML)."""
    out: list[str] = []
    for line in lines:
        s = C.shorten(line, width)
        if s:
            out.append(s)
    if len(out) > max_lines:
        out = out[: max_lines - 1] + [f"… (+{len(out) - max_lines + 1})"]
    text = "\n".join(out)
    return re.sub(r"(?i)href", lambda m: m.group(0)[:3] + "​" + m.group(0)[3:], text)


def material_indicators(fs: FeatureSet | None) -> dict[str, set[str]]:
    """Wskaźniki „śladu” materiału, które mogą łączyć materiały (do węzłów-wskaźników)."""
    if fs is None:
        return {}
    out: dict[str, set[str]] = defaultdict(set)
    inf, iocs = fs.infra, (fs.text.iocs or {})
    # „tracker:…” (keitaro, binom…) to rodzaj oprogramowania, a nie operator — słaba przesłanka, bez węzła-wskaźnika
    out["tracker"].update(v for v in inf.tracker_ids if not str(v).startswith("tracker:"))
    out["domain"].update(inf.domains)
    out["ip"].update(inf.ips)
    if inf.favicon_mmh3 is not None:
        out["favicon"].add(str(inf.favicon_mmh3))
    if inf.tls_fingerprint:
        out["tls"].add(inf.tls_fingerprint)
    out["form"].update(inf.form_actions)
    out["nameserver"].update(inf.nameservers)
    for key in ("crypto_btc", "crypto_eth", "crypto_trx"):
        out["wallet"].update(iocs.get(key, []))
    for key in ("phone", "email", "telegram", "whatsapp"):
        out[key].update(iocs.get(key, []))
    for img in fs.images:
        out["face"].update(f.identity for f in img.faces if f.identity)
    return {k: {str(v).strip() for v in vals if v and str(v).strip()} for k, vals in out.items() if vals}


def node_tooltip(mid: str, m: Material | None, fs: FeatureSet | None, campaign: Campaign | None, degree: int) -> str:
    lines = [mid]
    lines.append(f"Kampania: {campaign.id} — {campaign.name}" if campaign else "Kampania: brak (bez przypisania)")
    if m is not None:
        lines.append(f"Źródło: {C.relation_value(m.source)}" + (f" · {m.platform}" if m.platform else ""))
        if m.url:
            lines.append(f"URL: {m.url}")
        if m.advertiser:
            lines.append(f"Reklamodawca: {m.advertiser}")
        if m.observed_at:
            lines.append(f"Zaobserwowano: {C.fmt_dt(m.observed_at)}")
    if fs is not None:
        persons = list(
            dict.fromkeys([*fs.text.persons, *(f.identity for img in fs.images for f in img.faces if f.identity)])
        )
        if persons:
            lines.append("Osoby: " + C.join_values(persons, limit=5))
        if fs.text.narratives:
            lines.append("Narracje: " + C.join_values(fs.text.narratives, limit=4))
        if fs.infra.domains:
            lines.append("Domeny: " + C.join_values(fs.infra.domains, limit=4))
    lines.append(f"Powiązania w widoku: {degree}")
    return tooltip(lines)


def edge_tooltip(link: Link, label_a: str, label_b: str, n_evidence: int = 6) -> str:
    rel = C.relation_value(link.relation)
    lines = [
        f"{label_a}  ↔  {label_b}",
        f"{rel} · wynik {link.score:.1f} · pokrycie cech {link.coverage:.0%}",
        "Przesłanki:",
    ]
    for e in sorted(link.evidence, key=lambda x: -(x.weight or 0.0))[:n_evidence]:
        lines.append(f"• [{e.group}] {C.clean_text(e.detail)} (waga {e.weight:.1f})")
    if len(link.evidence) > n_evidence:
        lines.append(f"… oraz {len(link.evidence) - n_evidence} kolejnych")
    return tooltip(lines, max_lines=n_evidence + 5)


def _edge_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def build_graph(
    materials: Mapping[str, Material],
    features: Mapping[str, FeatureSet],
    links: Sequence[Link],
    campaigns: Sequence[Campaign],
    *,
    min_score: float = 0.0,
    relations: Iterable[str] | None = None,
    campaign_id: str | None = None,
    include_neighbors: bool = False,
    include_isolated: bool = True,
    hubs: bool = False,
    hub_kinds: Iterable[str] = DEFAULT_HUB_KINDS,
    max_hubs: int = 60,
    max_hub_fraction: float | None = None,
    max_edges: int = 3000,
    labels: bool | None = None,
    highlight: str | None = None,
    styles: Mapping[str, tuple[str, str]] | None = None,
) -> GraphData:
    """Dane grafu (węzły, krawędzie, legenda) po filtrach. Kolory kampanii nie zależą od filtrów."""
    styles = dict(styles) if styles is not None else C.campaign_styles(campaigns)
    camp_by_id = {c.id: c for c in campaigns}
    camp_of: dict[str, str] = {}
    for c in campaigns:
        for mid in c.members:
            camp_of.setdefault(mid, c.id)
    rel_ok = {str(r) for r in relations} if relations is not None else None

    selected = [
        lk
        for lk in links
        if lk.a != lk.b and lk.score >= min_score and (rel_ok is None or C.relation_value(lk.relation) in rel_ok)
    ]

    if campaign_id:
        members = set(camp_by_id[campaign_id].members) if campaign_id in camp_by_id else set()
        if include_neighbors:
            selected = [lk for lk in selected if lk.a in members or lk.b in members]
        else:
            selected = [lk for lk in selected if lk.a in members and lk.b in members]
        node_ids = set(members)
    else:
        node_ids = set(materials) if include_isolated else set()

    best: dict[tuple[str, str], Link] = {}
    for lk in selected:
        key = _edge_key(lk.a, lk.b)
        if key not in best or lk.score > best[key].score:
            best[key] = lk
    kept = sorted(best.values(), key=lambda x: (-x.score, x.a, x.b))
    truncated = len(kept) > max_edges
    kept = kept[:max_edges]
    for lk in kept:
        node_ids.add(lk.a)
        node_ids.add(lk.b)
    if highlight and highlight in materials:
        node_ids.add(highlight)

    degree: Counter[str] = Counter()
    for lk in kept:
        degree[lk.a] += 1
        degree[lk.b] += 1

    camp_order = {cid: i for i, cid in enumerate(sorted(camp_by_id))}
    ordered = sorted(node_ids, key=lambda mid: (camp_order.get(camp_of.get(mid, ""), 10**6), mid))
    show_labels = labels if labels is not None else len(ordered) <= 80

    data = GraphData(truncated=truncated)
    for mid in ordered:
        cid = camp_of.get(mid)
        color, shape = styles.get(cid, (C.UNASSIGNED_COLOR, "dot")) if cid else (C.UNASSIGNED_COLOR, "dot")
        m, fs = materials.get(mid), features.get(mid)
        node: dict[str, Any] = {
            "id": mid,
            "label": C.shorten(mid, 22) if show_labels else "",
            "title": node_tooltip(mid, m, fs, camp_by_id.get(cid) if cid else None, degree[mid]),
            "shape": shape,
            "color": color,
            "size": round(10 + min(14.0, 3.0 * math.sqrt(degree[mid])), 1),
            "borderWidth": 1,
        }
        if mid == highlight:
            node.update(size=node["size"] + 8, borderWidth=4, color={"background": color, "border": C.INK})
            node["label"] = C.shorten(mid, 22)
        data.nodes.append(node)

    for lk in kept:
        rel = C.relation_value(lk.relation)
        rel_color = C.RELATION_COLORS.get(rel, C.INK_MUTED)
        data.edges.append(
            {
                "from": lk.a,
                "to": lk.b,
                "width": round(1.0 + min(max(lk.score, 0.0), 24.0) / 4.0, 2),
                "color": {"color": rel_color, "highlight": HIGHLIGHT_COLOR, "hover": HIGHLIGHT_COLOR},
                "dashes": bool(C.RELATION_DASHES.get(rel, False)),
                "length": round(60 + 240 / (1.0 + max(lk.score, 0.0) / 4.0)),
                "title": edge_tooltip(lk, lk.a, lk.b),
            }
        )
    data.n_materials = len(ordered)
    data.n_links = len(kept)
    data.links = list(kept)

    if hubs and ordered:
        kinds = set(hub_kinds)
        per_value: dict[tuple[str, str], set[str]] = defaultdict(set)
        for mid in ordered:
            for kind, values in material_indicators(features.get(mid)).items():
                if kind in kinds:
                    for v in values:
                        per_value[(kind, v)].add(mid)
        cands = [(k, v, ms) for (k, v), ms in per_value.items() if len(ms) >= 2]
        if max_hub_fraction is not None and len(ordered) >= 6:
            limit = max(2, int(max_hub_fraction * len(ordered)))
            cands = [t for t in cands if len(t[2]) <= limit]
        cands.sort(key=lambda t: (-len(t[2]), t[0], t[1]))
        for kind, value, ms in cands[:max_hubs]:
            hid = f"hub:{kind}:{value}"
            data.nodes.append(
                {
                    "id": hid,
                    "label": f"{HUB_SHORT.get(kind, kind)}: {C.shorten(value, 28)}",
                    "title": tooltip(
                        [f"Wspólny wskaźnik — {HUB_KINDS.get(kind, kind)}", value, f"Materiały ({len(ms)}):"]
                        + sorted(ms)
                    ),
                    "shape": "diamond",
                    "color": HUB_COLOR,
                    "size": round(7 + min(12.0, 2.5 * math.sqrt(len(ms))), 1),
                    "borderWidth": 1,
                    "mass": 2,
                    "font": {"color": C.INK, "size": 10},
                }
            )
            for mid in sorted(ms):
                data.edges.append(
                    {
                        "from": mid,
                        "to": hid,
                        "width": 1,
                        "color": {"color": HUB_EDGE_COLOR, "highlight": HIGHLIGHT_COLOR, "hover": HIGHLIGHT_COLOR},
                        "dashes": [2, 4],
                        "length": 70,
                        "title": tooltip([f"{mid} → {HUB_KINDS.get(kind, kind)}", value]),
                    }
                )
            data.n_hubs += 1

    data.legend = _legend(ordered, camp_of, camp_by_id, styles, kept, data.n_hubs)
    return data


def _legend(
    ordered: Sequence[str],
    camp_of: Mapping[str, str],
    camp_by_id: Mapping[str, Campaign],
    styles: Mapping[str, tuple[str, str]],
    kept: Sequence[Link],
    n_hubs: int,
    max_campaigns: int = 12,
) -> list[dict[str, Any]]:
    present = Counter(camp_of[m] for m in ordered if m in camp_of)
    legend: list[dict[str, Any]] = []
    for cid in sorted(present)[:max_campaigns]:
        color, shape = styles.get(cid, (C.UNASSIGNED_COLOR, "dot"))
        name = camp_by_id[cid].name if cid in camp_by_id else ""
        legend.append(
            {
                "type": "campaign",
                "color": color,
                "shape": shape,
                "label": f"{cid} · {C.shorten(name, 40)} ({present[cid]})",
            }
        )
    if len(present) > max_campaigns:
        legend.append({"type": "more", "label": f"+{len(present) - max_campaigns} kolejnych kampanii"})
    n_unassigned = sum(1 for m in ordered if m not in camp_of)
    if n_unassigned:
        legend.append(
            {"type": "campaign", "color": C.UNASSIGNED_COLOR, "shape": "dot", "label": f"bez kampanii ({n_unassigned})"}
        )
    rels = Counter(C.relation_value(lk.relation) for lk in kept)
    for rel in C.RELATIONS:
        if rels.get(rel):
            legend.append(
                {
                    "type": "relation",
                    "color": C.RELATION_COLORS[rel],
                    "dashes": C.RELATION_DASHES[rel],
                    "label": f"{rel} ({rels[rel]})",
                }
            )
    if n_hubs:
        legend.append({"type": "hub", "color": HUB_COLOR, "shape": "diamond", "label": f"wspólny wskaźnik ({n_hubs})"})
    return legend


# ---------------------------------------------------------------- HTML

_CSS = """
<style>
  html, body { margin: 0; padding: 0; background: %(bg)s; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .card { border: none !important; }
  #mynetwork { border: none !important; float: none !important; }
  div.vis-tooltip {
    white-space: pre-wrap !important; max-width: 460px; font-family: system-ui, -apple-system, "Segoe UI", sans-serif !important;
    font-size: 12px !important; line-height: 1.4; color: %(ink)s !important; background: #ffffff !important;
    border: 1px solid rgba(11,11,11,0.12) !important; border-radius: 6px !important; box-shadow: 0 4px 14px rgba(0,0,0,0.12) !important;
    padding: 6px 9px !important;
  }
  #slad-legend {
    position: absolute; top: 8px; left: 8px; z-index: 10; max-width: 330px; max-height: 70%%; overflow: auto;
    background: rgba(252,252,251,0.94); border: 1px solid rgba(11,11,11,0.10); border-radius: 8px;
    padding: 4px 10px 6px; font-size: 12px; color: %(ink)s;
  }
  #slad-legend summary { cursor: pointer; font-weight: 600; color: %(ink2)s; padding: 2px 0; }
  #slad-legend .row { display: flex; align-items: center; gap: 6px; margin: 2px 0; white-space: nowrap; }
  #slad-legend .glyph { width: 16px; text-align: center; font-size: 14px; line-height: 1; }
  #slad-legend .line { display: inline-block; width: 22px; border-top-width: 3px; border-top-style: solid; }
  #slad-legend .hint { color: %(muted)s; margin-top: 4px; white-space: normal; }
  #slad-empty { padding: 40px; color: %(muted)s; text-align: center; }
</style>
<script>var highlightActive = false; var filterActive = false;</script>
"""

_AFTER_SCRIPT = """
<script>
(function () {
  try {
    if (typeof network !== "undefined" && network) {
      network.once("stabilizationIterationsDone", function () {
        network.setOptions({ physics: { enabled: false } });
        network.fit({ maxZoomLevel: 1.6, animation: false });
      });
    }
  } catch (e) {}
})();
</script>
"""


def legend_html(legend: Sequence[Mapping[str, Any]]) -> str:
    rows: list[str] = []
    for item in legend:
        label = _html.escape(str(item.get("label", "")))
        color = item.get("color", C.INK_MUTED)
        color = color if C.is_hex_color(str(color)) else C.INK_MUTED
        kind = item.get("type")
        if kind == "relation":
            style = "dashed" if item.get("dashes") else "solid"
            rows.append(
                f'<div class="row"><span class="line" style="border-top-color:{color};border-top-style:{style}"></span>{label}</div>'
            )
        elif kind == "more":
            rows.append(f'<div class="row hint">{label}</div>')
        else:
            glyph = C.SHAPE_GLYPHS.get(str(item.get("shape", "dot")), "●")
            rows.append(f'<div class="row"><span class="glyph" style="color:{color}">{glyph}</span>{label}</div>')
    hint = (
        '<div class="hint">Najedź na węzeł lub krawędź, aby zobaczyć szczegóły i przesłanki. '
        "Kliknięcie węzła wyróżnia jego sąsiadów.</div>"
    )
    return f'<div id="slad-legend"><details open><summary>Legenda</summary>{"".join(rows)}{hint}</details></div>'


def legend_chips_html(legend: Sequence[Mapping[str, Any]]) -> str:
    """Legenda w jednym zawijanym wierszu — do wyświetlenia nad grafem (nie zasłania węzłów)."""
    chips: list[str] = []
    for item in legend:
        label = _html.escape(str(item.get("label", "")))
        color = item.get("color", C.INK_MUTED)
        color = color if C.is_hex_color(str(color)) else C.INK_MUTED
        kind = item.get("type")
        if kind == "relation":
            style = "dashed" if item.get("dashes") else "solid"
            mark = (
                f'<span style="display:inline-block;width:20px;border-top:3px {style} {color};margin-right:5px"></span>'
            )
        elif kind == "more":
            chips.append(f'<span style="color:{C.INK_MUTED}">{label}</span>')
            continue
        else:
            glyph = C.SHAPE_GLYPHS.get(str(item.get("shape", "dot")), "●")
            mark = f'<span style="color:{color};font-size:15px;margin-right:4px">{glyph}</span>'
        chips.append(f'<span style="white-space:nowrap">{mark}{label}</span>')
    style = f"display:flex;flex-wrap:wrap;gap:4px 16px;align-items:center;font-size:0.85rem;color:{C.INK_2}"
    return f'<div style="{style}">{"".join(chips)}</div>'


def vis_options(n_nodes: int, n_edges: int, physics: bool = True) -> dict[str, Any]:
    return {
        "nodes": {
            "font": {
                "size": 11,
                "face": "system-ui, -apple-system, Segoe UI, sans-serif",
                "color": C.INK_2,
                "strokeWidth": 3,
                "strokeColor": C.SURFACE,
            },
            "borderWidthSelected": 3,
        },
        "edges": {"smooth": False, "selectionWidth": 2, "hoverWidth": 1.5},
        "interaction": {
            "hover": True,
            "tooltipDelay": 120,
            "hideEdgesOnDrag": n_edges > 800,
            "multiselect": True,
            "navigationButtons": False,
            "keyboard": False,
        },
        "physics": {
            "enabled": physics,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -55,
                "centralGravity": 0.012,
                "springLength": 100,
                "springConstant": 0.08,
                "damping": 0.55,
                "avoidOverlap": 0.5,
            },
            "maxVelocity": 50,
            "minVelocity": 1,
            "stabilization": {
                "enabled": True,
                "iterations": 500 if n_nodes <= 400 else 250,
                "updateInterval": 25,
                "fit": True,
            },
        },
        "layout": {"improvedLayout": n_nodes <= 150, "randomSeed": 7},
    }


_BOOTSTRAP = re.compile(
    r"<(?:link|script)\b[^>]*cdn\.jsdelivr\.net/npm/bootstrap[^>]*>(?:\s*</script>)?", re.IGNORECASE
)
_EMPTY_HEADING = re.compile(r"<center>\s*<h1>\s*</h1>\s*</center>", re.IGNORECASE)


def render_html(data: GraphData, height: int = 650, *, physics: bool = True, legend: bool = True) -> str:
    """Samodzielny HTML grafu (pyvis + vis-network inline, bez CDN). Wymaga pakietu ``pyvis``."""
    css = _CSS % {"bg": C.SURFACE, "ink": C.INK, "ink2": C.INK_2, "muted": C.INK_MUTED}
    if not data.nodes:
        return (
            f'<!doctype html><html><head><meta charset="utf-8">{css}</head><body>'
            '<div id="slad-empty">Brak materiałów do pokazania dla wybranych filtrów.</div></body></html>'
        )

    from pyvis.network import Network

    net = Network(
        height=f"{int(height)}px",
        width="100%",
        bgcolor=C.SURFACE,
        directed=False,
        neighborhood_highlight=True,
        cdn_resources="in_line",
    )
    for node in data.nodes:
        opts = dict(node)
        nid = opts.pop("id")
        label = opts.pop("label", "")
        shape = opts.pop("shape", "dot")
        color = opts.pop("color", C.UNASSIGNED_COLOR)
        net.add_node(nid, label=label or " ", shape=shape, color=color, **opts)
        net.node_map[nid]["label"] = label or ""
    # Krawędzie dopisujemy bezpośrednio (unikalne już po deduplikacji) — add_edge w pyvis ma koszt O(E²).
    known = set(net.node_ids)
    for edge in data.edges:
        if edge.get("from") in known and edge.get("to") in known:
            net.edges.append(dict(edge))
    net.options = vis_options(len(data.nodes), len(data.edges), physics=physics)

    page = net.generate_html()
    page = _BOOTSTRAP.sub("", page)
    page = _EMPTY_HEADING.sub("", page)
    page = page.replace("</head>", css + "</head>", 1)
    tail = (legend_html(data.legend) if legend and data.legend else "") + (_AFTER_SCRIPT if physics else "")
    if "</body>" in page:
        page = page.replace("</body>", tail + "</body>", 1)
    else:
        page += tail
    return page
