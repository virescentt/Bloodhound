"""Grupowanie materiałów w kampanie.

Metoda „graph” (domyślna):
1. graf: węzły = materiały, krawędzie = powiązania z wynikiem ≥ próg „powiązany”,
2. spójne składowe = kandydaci na kampanie,
3. składowe większe niż 6 węzłów są dzielone algorytmem Louvain (waga = wynik, rozdzielczość z konfiguracji,
   seed 42) — przeciw „łańcuchowaniu”: jedno słabe powiązanie nie skleja dwóch kampanii,
   przy czym pary duplikatów (relacja „duplikat” lub wynik ≥ próg „duplikat”) nigdy nie są rozdzielane,
4. części podziału są ponownie łączone, gdy łączy je gęsta sieć powiązań (≥ 30% możliwych par) albo
   co najmniej dwa silne powiązania (≥ próg „wariant”) — Louvain nie rozbija wariantów jednej kampanii.

Metoda „hdbscan” (opcjonalna): sklearn HDBSCAN na odległości 1/(1 + wynik) (brak powiązania = 1).
Wynik deterministyczny: kampanie od największej, przy remisie wg najmniejszego identyfikatora.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import networkx as nx
import numpy as np

from slad.models import Link, Relation
from slad.settings import LinkSettings

log = logging.getLogger(__name__)

SPLIT_MIN = 6  # składowe większe niż tyle węzłów są sprawdzane Louvainem
MERGE_DENSITY = 0.3
MERGE_STRONG = 2
HDBSCAN_MAX = 4000
LOUVAIN_SEED = 42


def _thr(cfg: LinkSettings) -> tuple[float, float, float]:
    t = cfg.thresholds or {}
    related = float(t.get("related", 4.0))
    variant = max(float(t.get("variant", 7.0)), related)
    duplicate = max(float(t.get("duplicate", 12.0)), variant)
    return related, variant, duplicate


class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def link_graph(ids: Iterable[str], links: Iterable[Link], cfg: LinkSettings) -> nx.Graph:
    """Graf powiązań ≥ próg „powiązany” (atrybuty krawędzi: score, dup)."""
    related, _, duplicate = _thr(cfg)
    nodes = sorted(set(ids))
    idset = set(nodes)
    best: dict[tuple[str, str], Link] = {}
    for lk in links:
        if lk.a == lk.b or lk.a not in idset or lk.b not in idset or lk.score < related:
            continue
        key = (lk.a, lk.b) if lk.a < lk.b else (lk.b, lk.a)
        if key not in best or lk.score > best[key].score:
            best[key] = lk
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for key in sorted(best):
        lk = best[key]
        G.add_edge(*key, score=float(lk.score), dup=bool(lk.relation == Relation.DUPLICATE or lk.score >= duplicate))
    return G


def _merge_back(H: nx.Graph, parts: list[list[str]], variant: float) -> list[list[str]]:
    parts = [sorted(p) for p in parts if p]
    while len(parts) > 1:
        where = {n: k for k, part in enumerate(parts) for n in part}
        inter: dict[tuple[int, int], list[float]] = {}
        for a, b, d in H.edges(data=True):
            ka, kb = where[a], where[b]
            if ka != kb:
                inter.setdefault((min(ka, kb), max(ka, kb)), []).append(float(d["score"]))
        best = None
        for (k1, k2), scores in sorted(inter.items()):
            n1, n2 = len(parts[k1]), len(parts[k2])
            density = len(scores) / float(n1 * n2)
            strong = sum(1 for s in scores if s >= variant)
            if min(n1, n2) == 1 or density >= MERGE_DENSITY or strong >= MERGE_STRONG:
                cand = (sum(scores), -k1, -k2)
                if best is None or cand > best:
                    best = cand
        if best is None:
            break
        k1, k2 = -best[1], -best[2]
        merged = sorted(parts[k1] + parts[k2])
        parts = [p for k, p in enumerate(parts) if k not in (k1, k2)] + [merged]
    return parts


def _split(H: nx.Graph, cfg: LinkSettings, variant: float) -> list[list[str]]:
    """Podział dużej składowej Louvainem (duplikaty ściągnięte do jednego węzła) + ponowne łączenie."""
    nodes = sorted(H.nodes)
    uf = _UnionFind(nodes)
    for a, b, d in H.edges(data=True):
        if d.get("dup"):
            uf.union(a, b)
    members: dict[str, list[str]] = {}
    for n in nodes:
        members.setdefault(uf.find(n), []).append(n)
    C = nx.Graph()
    C.add_nodes_from(sorted(members))
    for a, b, d in sorted(H.edges(data=True), key=lambda e: (e[0], e[1])):
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        if C.has_edge(ra, rb):
            C[ra][rb]["score"] += float(d["score"])
        else:
            C.add_edge(ra, rb, score=float(d["score"]))
    if C.number_of_nodes() <= 1 or C.number_of_edges() == 0:
        return [nodes]
    try:
        comms = nx.community.louvain_communities(
            C, weight="score", resolution=float(cfg.cluster.resolution or 1.0), seed=LOUVAIN_SEED
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Louvain nie zadziałał (%s) — składowa bez podziału", e)
        return [nodes]
    parts = [sorted(m for rep in comm for m in members[rep]) for comm in comms]
    return _merge_back(H, parts, variant)


def _hdbscan(G: nx.Graph, cfg: LinkSettings) -> list[list[str]] | None:
    nodes = sorted(n for n in G.nodes if G.degree(n) > 0)
    if len(nodes) < 2:
        return []
    if len(nodes) > HDBSCAN_MAX:
        log.warning("HDBSCAN: %d materiałów z powiązaniami — za dużo, używam metody graph", len(nodes))
        return None
    from sklearn.cluster import HDBSCAN

    pos = {n: i for i, n in enumerate(nodes)}
    D = np.ones((len(nodes), len(nodes)), dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    for a, b, d in G.edges(data=True):
        dist = 1.0 / (1.0 + float(d["score"]))
        D[pos[a], pos[b]] = D[pos[b], pos[a]] = dist
    min_size = max(2, int(cfg.cluster.min_size or 2))
    try:
        model = HDBSCAN(
            metric="precomputed", min_cluster_size=min_size, min_samples=1, allow_single_cluster=True, copy=True
        )
    except TypeError:  # starsze sklearn bez parametru copy
        model = HDBSCAN(metric="precomputed", min_cluster_size=min_size, min_samples=1, allow_single_cluster=True)
    labels = model.fit_predict(D)
    uf = _UnionFind(nodes)
    groups: dict[int, list[str]] = {}
    for n, lab in zip(nodes, labels.tolist(), strict=True):
        if lab >= 0:
            groups.setdefault(int(lab), []).append(n)
    for members in groups.values():
        for m in members[1:]:
            uf.union(members[0], m)
    for a, b, d in G.edges(data=True):  # duplikatów nie rozdzielamy
        if d.get("dup"):
            uf.union(a, b)
    out: dict[str, list[str]] = {}
    for n in nodes:  # szum HDBSCAN bez duplikatów zostaje singletonem (pomijany)
        out.setdefault(uf.find(n), []).append(n)
    return [sorted(v) for v in out.values() if len(v) >= 2]


def cluster(ids: Iterable[str], links: Iterable[Link], cfg: LinkSettings) -> list[list[str]]:
    """Kampanie jako listy identyfikatorów (≥ cfg.cluster.min_size, bez singletonów), deterministycznie."""
    if hasattr(cfg, "link") and not isinstance(cfg, LinkSettings):  # przekazano całe Settings
        cfg = cfg.link  # type: ignore[assignment]
    links = list(links)
    _, variant, _ = _thr(cfg)
    G = link_graph(ids, links, cfg)
    clusters: list[list[str]] | None = None
    if str(cfg.cluster.method or "graph").lower() == "hdbscan":
        try:
            clusters = _hdbscan(G, cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("HDBSCAN nie zadziałał (%s) — używam metody graph", e)
            clusters = None
    if clusters is None:
        clusters = []
        for comp in nx.connected_components(G):
            if len(comp) < 2:
                continue
            if len(comp) > SPLIT_MIN:
                clusters.extend(_split(G.subgraph(comp).copy(), cfg, variant))
            else:
                clusters.append(sorted(comp))
    min_size = max(2, int(cfg.cluster.min_size or 2))
    result = [sorted(c) for c in clusters if len(c) >= min_size]
    result.sort(key=lambda c: (-len(c), c[0]))
    return result
