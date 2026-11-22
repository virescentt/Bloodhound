"""Kandydaci do porównania: zamiast wszystkich N² par — tylko pary, które mogą być powiązane.

Do 400 materiałów porównujemy wszystkie pary (dokładnie i wciąż szybko). Powyżej — suma kandydatów z:
- indeksu odwrotnego wskaźników twardych (domena, tracker, telefon, portfel, favicon, IP…); wartości obecne
  w więcej niż max(50, 20% zbioru) materiałach są pomijane (wspólna infrastruktura, nie ślad kampanii) —
  z wyjątkiem przesłanek silnych (STRONG_KINDS: identyfikatory operatora — telefon, e-mail, tracker, portfel,
  Telegram, WhatsApp, IBAN, certyfikat TLS, skrócony link — oraz domena docelowa / z certyfikatu). Duży kubełek
  takiej wartości nie jest pomijany: do STRONG_BUCKET_MAX (= 400) materiałów — wszystkie pary (jak przy
  porównaniu wszystkich par małego zbioru, najwyżej 79 800 par), powyżej — „gwiazda”: każdy materiał kubełka
  w parze z jego pierwszym materiałem (k − 1 par, praca O(k)). Kampania większa niż 20% zbioru nadal tworzy
  spójną składową, a wagę przesłanki i tak obniża rzadkość (stats.py: identyfikatory operatora bez kary
  do 35% zbioru, domena do 25%),
- LSH MinHash (datasketch, próg ≈ 0.5) — teksty niemal identyczne,
- pHash: dokładne wyszukiwanie par o odległości Hamminga ≤ 10 (blokowo w numpy, popcount) + identyczne
  SHA-256 i dHash; simhash DOM ≤ 6 i identyczny HTML,
- twarze: top-10 najbliższych materiałów wg kosinusa embeddingów ≥ próg,
- TF-IDF: top-10 sąsiadów wg kosinusa (dla dużych zbiorów: wiersze przycięte do 160 najsilniejszych
  n-gramów — szybkie i z wysoką czułością; dokładny kosinus liczy potem porównanie pary).
Przesłanki miękkie (osoba, narracja, marka) nie generują kandydatów — same i tak nie wiążą materiałów.

Nowy materiał (neighbours) — te same źródła i ta sama reguła: z dużego kubełka przesłanki silnej bierzemy
STRONG_BUCKET_MAX pierwszych materiałów (w tym środek gwiazdy) — koszt analizy nowego materiału pozostaje
ograniczony także dla kampanii liczących tysiące materiałów.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from slad.link.evidence import DOM_MAX, MIN_MINHASH_WORDS, PHASH_MAX, TFIDF_MIN, LinkContext, Prepared
from slad.link.indicators import INDEX_KINDS, STRONG_KINDS
from slad.models import FeatureSet

log = logging.getLogger(__name__)

ALL_PAIRS_MAX = 400
INDEX_DF_CAP = 50
INDEX_DF_FRACTION = 0.2
# kubełek przesłanki silnej większy niż df_cap: do tylu materiałów — wszystkie pary, powyżej — gwiazda wokół
# pierwszego materiału; nowy materiał porównujemy z najwyżej tyloma pierwszymi materiałami kubełka
STRONG_BUCKET_MAX = ALL_PAIRS_MAX
LSH_THRESHOLD = 0.5
MINHASH_CANDIDATE = 0.45
TFIDF_TOPK = 10
TFIDF_PRUNE = 160
TFIDF_APPROX_MIN = 0.2  # próg na przyciętych wektorach (dokładny kosinus liczony później)
FACE_TOPK = 10
BLOCK = 512

Pair = tuple[str, str]


def _pair(a: str, b: str) -> Pair:
    return (a, b) if a < b else (b, a)


def _index_key(kind: str, value: str) -> tuple[str, str]:
    # domena z certyfikatu TLS trafia do tej samej przestrzeni co domeny — dopasowanie krzyżowe
    return ("domain" if kind == "cert_domain" else kind, value)


@dataclass
class BaseIndex:
    """Struktury wyszukiwania kandydatów dla materiałów bazy (budowane raz na LinkContext)."""

    ids: list[str]
    pos: dict[str, int]
    inv: dict[tuple[str, str], list[int]] = field(default_factory=dict)
    img_owner: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    img_hash: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint64))
    sha: dict[str, list[int]] = field(default_factory=dict)
    dhash: dict[int, list[int]] = field(default_factory=dict)
    dom_owner: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    dom_hash: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint64))
    html: dict[str, list[int]] = field(default_factory=dict)
    face_owner: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    face_mat: np.ndarray | None = None
    mh_owner: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    mh_mat: np.ndarray | None = None

    @property
    def df_cap(self) -> int:
        return max(INDEX_DF_CAP, int(INDEX_DF_FRACTION * len(self.ids)))


def base_index(ctx: LinkContext) -> BaseIndex:
    """Indeks kandydatów bazy kontekstu (z pamięcią podręczną w kontekście)."""
    cached = getattr(ctx, "_base_index", None)
    if cached is not None and cached.ids == ctx.ids:
        return cached
    idx = _build_index(ctx)
    ctx._base_index = idx  # type: ignore[attr-defined]
    return idx


def _build_index(ctx: LinkContext) -> BaseIndex:
    ids = list(ctx.ids)
    bi = BaseIndex(ids=ids, pos={mid: i for i, mid in enumerate(ids)})
    img_owner, img_hash, dom_owner, dom_hash = [], [], [], []
    face_blocks: list[tuple[int, np.ndarray]] = []
    mh_rows, mh_owner = [], []
    for i, mid in enumerate(ids):
        p = ctx.prep[mid]
        for kind in INDEX_KINDS:
            for v in p.ind.get(kind, ()):
                bi.inv.setdefault(_index_key(kind, v), []).append(i)
        for ph, dh, sha, _ in p.images:
            if ph is not None:
                img_owner.append(i)
                img_hash.append(ph)
            if sha:
                bi.sha.setdefault(sha, []).append(i)
            if dh is not None:
                bi.dhash.setdefault(dh, []).append(i)
        if p.dom is not None:
            dom_owner.append(i)
            dom_hash.append(p.dom)
        if p.html_sha:
            bi.html.setdefault(p.html_sha, []).append(i)
        if p.faces is not None:
            face_blocks.append((i, p.faces))
        if p.minhash is not None and p.n_words >= MIN_MINHASH_WORDS:
            mh_rows.append(p.minhash)
            mh_owner.append(i)
    bi.img_owner = np.asarray(img_owner, dtype=np.int64)
    bi.img_hash = np.asarray(img_hash, dtype=np.uint64)
    bi.dom_owner = np.asarray(dom_owner, dtype=np.int64)
    bi.dom_hash = np.asarray(dom_hash, dtype=np.uint64)
    if face_blocks:
        dims = [m.shape[1] for _, m in face_blocks]
        dim = max(set(dims), key=dims.count)
        kept = [(o, m) for o, m in face_blocks if m.shape[1] == dim]
        bi.face_mat = np.vstack([m for _, m in kept]).astype(np.float32)
        bi.face_owner = np.asarray([o for o, m in kept for _ in range(m.shape[0])], dtype=np.int64)
    if mh_rows:
        lens = [len(r) for r in mh_rows]
        length = max(set(lens), key=lens.count)
        keep = [k for k, r in enumerate(mh_rows) if len(r) == length]
        bi.mh_mat = np.vstack([mh_rows[k] for k in keep])
        bi.mh_owner = np.asarray([mh_owner[k] for k in keep], dtype=np.int64)
    return bi


# ---------------------------------------------------------------- generatory par (cała baza)


def _index_pairs(bi: BaseIndex, keep: set[int] | None = None) -> set[Pair]:
    """Pary z indeksu odwrotnego wskaźników.

    Kubełek ≤ df_cap materiałów → wszystkie pary. Większy kubełek wartości przesłanki silnej (STRONG_KINDS):
    ≤ STRONG_BUCKET_MAX materiałów → nadal wszystkie pary; jeszcze większy → gwiazda (każdy materiał w parze
    z pierwszym materiałem kubełka: k − 1 par, praca O(k) zamiast O(k²)). Większe kubełki pozostałych rodzajów
    są pomijane (wspólna infrastruktura). `keep` (pozycje w bazie) — gdy candidate_pairs dostało tylko część
    bazy, duży kubełek jest najpierw zawężany do niej (środek gwiazdy jest wtedy porównywanym materiałem).
    """
    out: set[Pair] = set()
    cap = bi.df_cap
    for (kind, _), members in bi.inv.items():
        uniq = sorted(set(members))
        if len(uniq) <= cap:
            head, tail = uniq, []
        elif kind in STRONG_KINDS:
            if keep is not None:
                uniq = [i for i in uniq if i in keep]
            # powyżej limitu sama gwiazda — klikę z doczepionymi liśćmi Louvain rozdzieliłby na dwie kampanie
            head, tail = (uniq, []) if len(uniq) <= STRONG_BUCKET_MAX else (uniq[:1], uniq[1:])
        else:
            continue
        for i, j in combinations(head, 2):
            out.add(_pair(bi.ids[i], bi.ids[j]))
        if head and tail:
            hub = bi.ids[head[0]]
            out.update(_pair(hub, bi.ids[j]) for j in tail)
    return out


def _bucket_neighbours(members: list[int], kind: str, cap: int) -> list[int]:
    """Materiały kubełka indeksu do porównania z nowym materiałem — ta sama reguła co w _index_pairs:
    kubełek ≤ cap → cały; większy → przesłanka silna: STRONG_BUCKET_MAX pierwszych (ze środkiem gwiazdy),
    inny rodzaj: nic."""
    uniq = sorted(set(members))
    if len(uniq) <= cap:
        return uniq
    if kind in STRONG_KINDS:
        return uniq[:STRONG_BUCKET_MAX]
    return []


def _bucket_pairs(bi: BaseIndex, buckets: Iterable[list[int]], cap: int) -> set[Pair]:
    out: set[Pair] = set()
    for members in buckets:
        uniq = sorted(set(members))
        if 2 <= len(uniq) <= cap:
            for i, j in combinations(uniq, 2):
                out.add(_pair(bi.ids[i], bi.ids[j]))
    return out


def _hamming_pairs(owner: np.ndarray, hashes: np.ndarray, max_dist: int, ids: list[str]) -> set[Pair]:
    """Wszystkie pary hashy 64-bit o odległości Hamminga ≤ max_dist (blokowo, górny trójkąt)."""
    out: set[Pair] = set()
    m = len(hashes)
    if m < 2:
        return out
    for s in range(0, m, BLOCK):
        e = min(m, s + BLOCK)
        d = np.bitwise_count(hashes[s:e, None] ^ hashes[None, s:])
        bi_, bj_ = np.nonzero(d <= max_dist)
        for a, b in zip((bi_ + s).tolist(), (bj_ + s).tolist(), strict=True):
            if b <= a:
                continue
            oa, ob = int(owner[a]), int(owner[b])
            if oa != ob:
                out.add(_pair(ids[oa], ids[ob]))
    return out


def _face_pairs(bi: BaseIndex, threshold: float) -> set[Pair]:
    out: set[Pair] = set()
    if bi.face_mat is None or len(bi.face_mat) < 2:
        return out
    E, owner = bi.face_mat, bi.face_owner
    for s in range(0, len(E), BLOCK):
        sims = E[s : s + BLOCK] @ E.T
        rows_owner = owner[s : s + BLOCK]
        sims[owner[None, :] == rows_owner[:, None]] = -1.0
        k = min(FACE_TOPK * 2, sims.shape[1])
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        for r in range(sims.shape[0]):
            seen: set[int] = set()
            for c in sorted(top[r].tolist(), key=lambda c: -sims[r, c]):
                if sims[r, c] < threshold or len(seen) >= FACE_TOPK:
                    break
                o = int(owner[c])
                if o not in seen:
                    seen.add(o)
                    out.add(_pair(bi.ids[int(rows_owner[r])], bi.ids[o]))
    return out


def _minhash_pairs(ctx: LinkContext, bi: BaseIndex) -> set[Pair]:
    out: set[Pair] = set()
    if bi.mh_mat is None or len(bi.mh_mat) < 2:
        return out
    owners = bi.mh_owner.tolist()
    num_perm = bi.mh_mat.shape[1]
    try:
        from datasketch import MinHash, MinHashLSH

        lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=num_perm)
        perms = MinHash(num_perm=num_perm, scheme="legacy").permutations  # wspólne (nieużywane: bez update())
        objs = []
        for k, o in enumerate(owners):
            mh = MinHash(num_perm=num_perm, hashvalues=bi.mh_mat[k], scheme="legacy", permutations=perms)
            lsh.insert(o, mh, check_duplication=False)
            objs.append(mh)
        for k, o in enumerate(owners):
            for other in lsh.query(objs[k]):
                if other != o:
                    out.add(_pair(bi.ids[o], bi.ids[int(other)]))
        return out
    except Exception as e:  # noqa: BLE001 — awaryjnie własne pasma LSH
        log.info("datasketch LSH niedostępny (%s) — własne pasma", e)
    rows = 5
    bands = max(1, num_perm // rows)
    for b in range(bands):
        buckets: dict[bytes, list[int]] = {}
        sl = bi.mh_mat[:, b * rows : (b + 1) * rows]
        for k, o in enumerate(owners):
            buckets.setdefault(sl[k].tobytes(), []).append(o)
        out |= _bucket_pairs(bi, buckets.values(), cap=10**9)
    return out


def _prune_rows(X, k: int):
    """Każdy wiersz CSR przycięty do k największych wag i ponownie znormalizowany (L2)."""
    import scipy.sparse as sp

    X = X.tocsr()
    indptr, indices, data = X.indptr, X.indices, X.data
    new_ptr = [0]
    new_idx: list[np.ndarray] = []
    new_dat: list[np.ndarray] = []
    for i in range(X.shape[0]):
        s, e = indptr[i], indptr[i + 1]
        d = data[s:e]
        if e - s > k:
            top = np.argpartition(-d, k - 1)[:k]
            ii, dd = indices[s:e][top], d[top]
        else:
            ii, dd = indices[s:e], d
        norm = float(np.sqrt(np.dot(dd, dd))) or 1.0
        new_idx.append(ii)
        new_dat.append((dd / norm).astype(np.float32))
        new_ptr.append(new_ptr[-1] + len(ii))
    if not new_idx:
        return sp.csr_matrix(X.shape, dtype=np.float32)
    return sp.csr_matrix(
        (np.concatenate(new_dat), np.concatenate(new_idx), np.asarray(new_ptr)), shape=X.shape, dtype=np.float32
    )


def _topk_from_dense(S: np.ndarray, ids: list[str], min_sim: float) -> set[Pair]:
    out: set[Pair] = set()
    n = S.shape[0]
    if n < 2:
        return out
    k = min(TFIDF_TOPK + 1, n)
    for s in range(0, n, BLOCK):
        blk = np.array(S[s : s + BLOCK], dtype=np.float32, copy=True)
        rows = np.arange(blk.shape[0])
        blk[rows, rows + s] = -1.0
        top = np.argpartition(-blk, k - 1, axis=1)[:, :k]
        for r in range(blk.shape[0]):
            for c in top[r].tolist():
                if blk[r, c] >= min_sim:
                    out.add(_pair(ids[s + r], ids[c]))
    return out


def _tfidf_pairs(ctx: LinkContext) -> set[Pair]:
    out: set[Pair] = set()
    if ctx.X is None:
        return out
    ids = ctx.ids
    if ctx.S is not None:
        out |= _topk_from_dense(ctx.S, ids, TFIDF_MIN)
        out |= _topk_from_dense(ctx.Sc, ids, TFIDF_MIN)
        return out
    for M in (ctx.X, ctx.Xc):
        P = _prune_rows(M, TFIDF_PRUNE)
        PT = P.T.tocsr()
        for s in range(0, P.shape[0], BLOCK):
            S = (P[s : s + BLOCK] @ PT).tocsr()
            for r in range(S.shape[0]):
                a, b = S.indptr[r], S.indptr[r + 1]
                if a == b:
                    continue
                cols, vals = S.indices[a:b], S.data[a:b]
                order = np.argsort(-vals)[: TFIDF_TOPK + 1]
                for c, v in zip(cols[order].tolist(), vals[order].tolist(), strict=True):
                    if c != s + r and v >= TFIDF_APPROX_MIN:
                        out.add(_pair(ids[s + r], ids[c]))
    return out


def candidate_pairs(features: Iterable[FeatureSet], ctx: LinkContext) -> set[Pair]:
    """Pary (a < b) materiałów do porównania. Materiały spoza bazy kontekstu są pomijane (zob. neighbours)."""
    ids = []
    seen: set[str] = set()
    for fs in features:
        mid = fs.material_id
        if mid in seen:
            continue
        seen.add(mid)
        if mid in ctx.prep:
            ids.append(mid)
        else:
            log.warning("Materiał %s spoza kontekstu — pomijam przy generowaniu kandydatów", mid)
    if len(ids) < 2:
        return set()
    if len(ids) <= ALL_PAIRS_MAX:
        ids.sort()
        return set(combinations(ids, 2))
    bi = base_index(ctx)
    keep_pos = {bi.pos[mid] for mid in ids} if len(ids) < len(ctx.ids) else None
    pairs: set[Pair] = set()
    generators: list[tuple[str, Callable[[], set[Pair]]]] = [
        ("indeks wskaźników", lambda: _index_pairs(bi, keep_pos)),
        ("MinHash LSH", lambda: _minhash_pairs(ctx, bi)),
        ("pHash", lambda: _hamming_pairs(bi.img_owner, bi.img_hash, PHASH_MAX, bi.ids)),
        ("SHA-256 / dHash", lambda: _bucket_pairs(bi, list(bi.sha.values()) + list(bi.dhash.values()), bi.df_cap)),
        ("simhash DOM", lambda: _hamming_pairs(bi.dom_owner, bi.dom_hash, DOM_MAX, bi.ids)),
        ("HTML", lambda: _bucket_pairs(bi, bi.html.values(), bi.df_cap)),
        ("twarze", lambda: _face_pairs(bi, float(ctx.cfg.faces.match_threshold))),
        ("TF-IDF", lambda: _tfidf_pairs(ctx)),
    ]
    for name, gen in generators:
        try:
            found = gen()
            log.debug("Kandydaci %s: %d par", name, len(found))
            pairs |= found
        except Exception as e:  # noqa: BLE001 — jeden generator nie zatrzymuje pozostałych
            log.warning("Generator kandydatów %s nie zadziałał: %s", name, e)
    if len(ids) < len(ctx.ids):
        keep = set(ids)
        pairs = {p for p in pairs if p[0] in keep and p[1] in keep}
    return pairs


# ---------------------------------------------------------------- jeden materiał vs baza (analiza nowego)


def neighbours(p: Prepared, ctx: LinkContext) -> set[str]:
    """Materiały bazy, z którymi warto porównać materiał `p` (np. nowy materiał dołączony przez ctx.add)."""
    bi = base_index(ctx)
    n = len(bi.ids)
    if n == 0:
        return set()
    if n <= ALL_PAIRS_MAX:
        return {mid for mid in bi.ids if mid != p.id}
    out: set[int] = set()
    cap = bi.df_cap
    try:
        for kind in INDEX_KINDS:
            for v in p.ind.get(kind, ()):
                key = _index_key(kind, v)
                out.update(_bucket_neighbours(bi.inv.get(key, []), key[0], cap))
        if "cert_domain" in p.ind:  # domeny z certyfikatu nowego materiału vs domeny bazy
            for v in p.ind["cert_domain"]:
                out.update(_bucket_neighbours(bi.inv.get(("domain", v), []), "domain", cap))
    except Exception as e:  # noqa: BLE001
        log.warning("Kandydaci (indeks) dla %s: %s", p.id, e)
    try:
        for ph, dh, sha, _ in p.images:
            if sha:
                out.update(bi.sha.get(sha, []))
            if dh is not None:
                out.update(bi.dhash.get(dh, []))
            if ph is not None and len(bi.img_hash):
                d = np.bitwise_count(bi.img_hash ^ np.uint64(ph))
                out.update(bi.img_owner[d <= PHASH_MAX].tolist())
        if p.html_sha:
            out.update(bi.html.get(p.html_sha, []))
        if p.dom is not None and len(bi.dom_hash):
            d = np.bitwise_count(bi.dom_hash ^ np.uint64(p.dom))
            out.update(bi.dom_owner[d <= DOM_MAX].tolist())
    except Exception as e:  # noqa: BLE001
        log.warning("Kandydaci (hashe) dla %s: %s", p.id, e)
    try:
        if p.minhash is not None and bi.mh_mat is not None and p.minhash.shape[0] == bi.mh_mat.shape[1]:
            j = np.count_nonzero(bi.mh_mat == p.minhash[None, :], axis=1) / float(bi.mh_mat.shape[1])
            out.update(bi.mh_owner[j >= MINHASH_CANDIDATE].tolist())
    except Exception as e:  # noqa: BLE001
        log.warning("Kandydaci (MinHash) dla %s: %s", p.id, e)
    try:
        if p.faces is not None and bi.face_mat is not None and p.faces.shape[1] == bi.face_mat.shape[1]:
            sims = (bi.face_mat @ p.faces.T).max(axis=1)
            order = np.argsort(-sims)
            taken: set[int] = set()
            thr = float(ctx.cfg.faces.match_threshold)
            for r in order.tolist():
                if sims[r] < thr or len(taken) >= FACE_TOPK:
                    break
                taken.add(int(bi.face_owner[r]))
            out |= taken
    except Exception as e:  # noqa: BLE001
        log.warning("Kandydaci (twarze) dla %s: %s", p.id, e)
    try:
        sims = ctx.base_text_sims(p)
        if sims is not None:
            for vec in sims:
                if len(vec) == 0:
                    continue
                k = min(TFIDF_TOPK, len(vec))
                top = np.argpartition(-vec, k - 1)[:k]
                out.update(int(i) for i in top.tolist() if vec[i] >= TFIDF_MIN)
    except Exception as e:  # noqa: BLE001
        log.warning("Kandydaci (TF-IDF) dla %s: %s", p.id, e)
    return {bi.ids[i] for i in out if bi.ids[i] != p.id}
