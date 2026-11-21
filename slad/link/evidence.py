"""Porównanie pary materiałów: przesłanki (Evidence) z wagami → wynik → relacja (Link).

Model: wynik = Σ wag przesłanek (≈ log-iloraz szans „ta sama kampania”). Każda przesłanka ma rodzaj (klucz
wagi z config/settings.yaml → link.weights), grupę, opis po polsku i wkład w wynik. Suma wkładów = wynik.

Przesłanki:
- wskaźniki dokładne (indicators.py): waga rodzaju × rzadkość wartości (stats.py); w rodzaju liczą się
  najwyżej 2 najsilniejsze wspólne wartości (druga z połową wagi); słowa kluczowe łącznie ≤ 1.0,
- tekst: Jaccard sygnatur MinHash (niemal identyczny tekst) + kosinus TF-IDF n-gramów znakowych 3–5
  (całość tekstu oraz sama treść reklamy: wejście + OCR — odporne na brak strony docelowej i szum OCR);
  słabsza z dwóch miar liczy się w połowie (bez podwójnego liczenia),
- obraz: najbliższa para obrazów — identyczny plik (SHA-256) albo odległość Hamminga pHash ≤ 10/64,
  opcjonalnie CLIP; podobny układ przy zupełnie innym tekście reklamy (cos < 0.15) to raczej wspólny
  szablon — połowa wagi i przesłanka średnia,
- twarze: najwyższy kosinus embeddingów SFace ≥ progu z konfiguracji,
- szablon strony: odległość simhash DOM ≤ 6 (lub identyczny HTML); tylko strony nie-generyczne,
- rejestrator + data rejestracji domen (±3 dni) dla różnych domen.

Rzadkość: wartości dokładne — częstość w korpusie (stats.py); obraz i szablon strony — gęstość sąsiedztwa
(ile innych materiałów ma obraz / szablon w promieniu Δ ≤ 6): wspólny układ zrzutów platformy waży mało.

Trzy poziomy przesłanek i ochrona przed fałszywymi powiązaniami (ta sama twarz celebryty występuje
w wielu kampaniach, „tło internetu” — CDN, hosting, skracacze — w niezależnych materiałach):
- silne (identyfikatory operatora: piksel/tracker, portfel, telefon, e-mail, Telegram/WhatsApp, IBAN, certyfikat
  TLS, domena docelowa, skrócony link; tekst niemal identyczny; ta sama grafika pHash Δ ≤ 6; identyczny HTML),
- średnie (host, domena pośrednia, favicona, IP, tytuł strony, formularz, szablon DOM, podobny tekst…) —
  powiązanie wymaga przesłanki silnej albo ≥ 2 różnych rodzajów średnich; liczą się tylko wartości nieczęste
  (rzadkość ≥ 0.75), inaczej wynik zostaje poniżej progu,
- miękkie (osoby, marka, narracja, słowa kluczowe, ASN, DNS, rejestrator, CLIP) łącznie
  ≤ min(0.875 × próg, max(0.5 × próg, 2 × suma przesłanek twardych)) — same nigdy nie tworzą powiązania;
  osoby (nazwisko + twarz) łącznie ≤ 0.75 × próg „powiązany”.

Relacja: duplikat (treść niemal identyczna we wszystkich grupach treści obecnych w obu materiałach: tekst
Jaccard ≥ 0.9 lub cos ≥ 0.98, obraz — ten sam plik lub pHash/dHash Δ ≤ 2) · wariant (wynik ≥ próg „wariant”)
· powiązany (wynik ≥ próg). Wysoki wynik z samej wspólnej infrastruktury to wariant, nie duplikat.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from slad.link.indicators import (
    INDICATOR_KINDS,
    KIND_GROUP,
    MAX_VALUES,
    SOFT_KINDS,
    STRONG_KINDS,
    Stoplists,
    display_value,
    fold,
    indicators,
    load_stoplists,
    page_is_generic,
    registrar_is_common,
    weight,
)
from slad.link.stats import (
    continuous_values,
    document_frequency,
    hex64,
    rarity_factor,
    rarity_from_df,
    stats_from_values,
)
from slad.models import CorpusStats, Evidence, FeatureSet, Link, Relation
from slad.settings import Settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- parametry metody

MINHASH_MIN = 0.5  # Jaccard ≥ 0.5 → przesłanka „tekst niemal identyczny”
TFIDF_MIN = 0.35  # kosinus TF-IDF ≥ 0.35 → przesłanka „podobna treść”
TFIDF_GAMMA = 1.5  # krzywa wagi TF-IDF (niskie podobieństwa ważą niewiele)
PHASH_MAX = 10  # maks. odległość Hamminga pHash (z 64 bitów)
PHASH_SCALE = 12.0
DHASH_SUSPECT = 20  # pHash bliski, a dHash bardzo różny → połowa wagi
DOM_MAX = 6
DOM_SCALE = 8.0
CLIP_MIN = 0.88
FACE_FULL = 0.75  # kosinus twarzy, od którego przesłanka ma pełną wagę
REG_DAYS = 3
MIN_TEXT_CHARS = 20
MIN_MINHASH_WORDS = 6
KEYWORD_CAP = 1.0
PERSON_CAP = 0.75  # × próg „powiązany”
SOFT_CAP = 0.875  # × próg „powiązany”
SOFT_FLOOR = 0.5  # × próg „powiązany”
SOFT_HARD_RATIO = 2.0
CORROBORATION_CAP = 0.95  # × próg — bez przesłanki silnej i z < 2 rodzajami średnich wynik zostaje poniżej progu
RARE_ENOUGH = 0.75  # współczynnik rzadkości, od którego wartość może potwierdzać powiązanie
TFIDF_STRONG = 0.8  # kosinus treści, od którego podobieństwo tekstu jest przesłanką silną
TFIDF_CORROB = 0.7  # kosinus, od którego podobny tekst może potwierdzać inną przesłankę średnią
TEXT_FULL_CHARS = 120  # długość krótszego tekstu, od której przesłanki tekstowe mają pełną wagę
TEXT_STRONG_CHARS = 60  # krótszy tekst nie daje przesłanki silnej (szablonowe hasła reklamowe)
PHASH_STRONG = 6  # odległość pHash, do której grafika jest przesłanką silną
TEXT_CONTRADICTS = 0.15  # kosinus treści reklam poniżej progu: podobna grafika to raczej wspólny szablon
TEXT_DUP_JACCARD = 0.9  # duplikat: tekst niemal identyczny…
TEXT_DUP_COSINE = 0.98
IMAGE_DUP_DIST = 2  # …i ten sam plik albo pHash Δ ≤ 2 przy dHash Δ ≤ 2 (warianty tej samej makiety mają pHash ≤ 4)
IMAGE_DUP_DHASH = 2
RARITY_NOTE_BELOW = 0.9  # przy mniejszym współczynniku opis wspomina o obniżonej wadze
TFIDF_MAX_CHARS = 5000
DENSE_MAX = 1500  # do tej liczby materiałów macierz podobieństw tekstu liczona w całości
HASH_FEATURES = 2**21
MAX_EXTRA = 256  # ile materiałów spoza bazy (add) pamięta kontekst

_MASK64 = (1 << 64) - 1
_NON_WORD = re.compile(r"[^\w\s]+")


def thresholds(cfg: Settings) -> tuple[float, float, float]:
    """(powiązany, wariant, duplikat) z konfiguracji (z wartościami domyślnymi)."""
    t = cfg.link.thresholds or {}
    related = float(t.get("related", 4.0))
    variant = float(t.get("variant", 7.0))
    duplicate = float(t.get("duplicate", 12.0))
    return related, max(variant, related), max(duplicate, variant, related)


# ---------------------------------------------------------------- przygotowanie materiału


def _doc(text: str | None) -> str:
    """Tekst do TF-IDF: bez diakrytyków, małe litery, bez interpunkcji, ograniczona długość."""
    if not text:
        return ""
    t = fold(str(text)[: TFIDF_MAX_CHARS * 2])
    t = _NON_WORD.sub(" ", t)
    return " ".join(t.split())[:TFIDF_MAX_CHARS]


def _creative_doc(fs: FeatureSet) -> str:
    """Treść samej reklamy (tekst z wejścia + OCR) — bez strony docelowej."""
    d = _doc(f"{fs.text.input_text or ''} {fs.text.ocr_text or ''}")
    return d if len(d) >= MIN_TEXT_CHARS else ""


def _segment_docs(fs: FeatureSet) -> tuple[str, str]:
    """Kanały treści reklamy osobno: (tekst z wejścia, tekst z OCR obrazów).

    Ta sama treść reklamy bywa podana raz jako pole tekstowe, raz tylko jako zrzut ekranu; porównanie
    kanałów osobno (także na krzyż) jest odporne na materiały częściowe — złączenie wejścia z OCR
    „rozcieńcza” podobieństwo, gdy jeden z materiałów ma tylko jeden kanał."""
    out = []
    for raw in (fs.text.input_text, fs.text.ocr_text):
        d = _doc(raw)
        out.append(d if len(d) >= MIN_TEXT_CHARS else "")
    return out[0], out[1]


# pary kanałów (a, b) w kolejności macierzy podobieństw: wejście–wejście, wejście–OCR, OCR–wejście, OCR–OCR
_SEG_COMBOS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _int64(value: str | None, degenerate_check: bool = True) -> int | None:
    h = hex64(value)
    if h is None:
        return None
    v = int(h, 16)
    if degenerate_check:
        bits = v.bit_count()
        if bits <= 2 or bits >= 62:  # obraz pusty / jednolity — hash bez informacji
            return None
    return v


def _unit_rows(vectors: list[list[float]]) -> np.ndarray | None:
    vecs = [v for v in vectors if v]
    if not vecs:
        return None
    dim = Counter(len(v) for v in vecs).most_common(1)[0][0]
    try:
        m = np.asarray([v for v in vecs if len(v) == dim], dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(m)):
        return None
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    keep = norms[:, 0] > 1e-9
    if not keep.any():
        return None
    return m[keep] / norms[keep]


def _minhash_array(values: list[int]) -> np.ndarray | None:
    if not values or len(values) < 16:
        return None
    try:
        arr = np.fromiter((int(v) & _MASK64 for v in values), dtype=np.uint64, count=len(values))
    except (TypeError, ValueError, OverflowError):
        return None
    if np.all(arr == arr[0]):  # sygnatura pustego zbioru (krótki tekst) — bez informacji
        return None
    return arr


def _naive_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _norm_url(u: str) -> str:
    u = str(u).strip().lower()
    u = re.sub(r"^[a-z]+://", "", u)
    u = u.removeprefix("www.")
    return u.split("#")[0].rstrip("/")


@dataclass
class Prepared:
    """Materiał przygotowany do szybkich porównań (wskaźniki, hashe jako int, macierze embeddingów)."""

    id: str
    fs: FeatureSet
    ind: dict[str, frozenset[str]]
    generic: bool = False
    images: list[tuple[int | None, int | None, str, str | None]] = field(default_factory=list)  # pHash, dHash, sha, hex
    faces: np.ndarray | None = None
    clip: np.ndarray | None = None
    minhash: np.ndarray | None = None
    dom: int | None = None
    dom_hex: str | None = None
    html_sha: str | None = None
    n_chars: int = 0
    n_words: int = 0
    creative_chars: int = 0
    seg_chars: tuple[int, int] = (0, 0)  # długość kanałów treści reklamy: (wejście, OCR)
    registrar: str | None = None
    registrar_common: bool = False
    created: datetime | None = None
    available: frozenset[str] = frozenset()
    urls: frozenset[str] = frozenset()

    @property
    def has_content(self) -> bool:
        return self.n_chars >= MIN_TEXT_CHARS or bool(self.images)


def prepare(fs: FeatureSet, stop: Stoplists | None = None) -> Prepared:
    """Przygotowanie materiału; błąd w jednej grupie cech nie zatrzymuje pozostałych."""
    stop = stop or load_stoplists()
    try:
        ind = {k: frozenset(v) for k, v in indicators(fs, stop).items() if v}
    except Exception as e:  # noqa: BLE001
        log.warning("Wskaźniki materiału %s niedostępne: %s", fs.material_id, e)
        ind = {}
    p = Prepared(id=fs.material_id, fs=fs, ind=ind)
    try:
        p.generic = page_is_generic(fs, stop)
    except Exception:  # noqa: BLE001
        p.generic = False
    for img in fs.images:
        try:
            sha = (img.sha256 or "").strip().lower()
            p.images.append((_int64(img.phash), _int64(img.dhash), sha, hex64(img.phash)))
        except Exception:  # noqa: BLE001
            continue
    try:
        p.faces = _unit_rows([list(f.embedding) for img in fs.images for f in img.faces if f.embedding])
        p.clip = _unit_rows([list(img.clip) for img in fs.images if img.clip])
    except Exception:  # noqa: BLE001
        p.faces = p.clip = None
    p.minhash = _minhash_array(fs.text.minhash)
    text = fs.text.text or ""
    p.n_chars = len(text.strip())
    p.n_words = len(text.split())
    p.creative_chars = len(_creative_doc(fs))
    seg_in, seg_ocr = _segment_docs(fs)
    p.seg_chars = (len(seg_in), len(seg_ocr))
    inf = fs.infra
    if not p.generic:
        p.dom = _int64(inf.dom_simhash, degenerate_check=False)
        p.dom_hex = hex64(inf.dom_simhash)
        p.html_sha = (inf.html_sha256 or "").strip().lower() or None
    if inf.registrar:
        p.registrar = fold(inf.registrar) or None
        p.registrar_common = registrar_is_common(inf.registrar, stop)
    p.created = _naive_utc(inf.domain_created)
    p.available = frozenset(fs.available)
    urls = [*inf.input_urls]
    if inf.final_url:
        urls.append(inf.final_url)
    p.urls = frozenset(u for u in (_norm_url(x) for x in urls if x) if u)
    return p


def _reclassify(p: Prepared, redirectors: frozenset[str]) -> None:
    """Domeny znane w korpusie jako pośrednie (przekierowania) → rodzaj „redirect_domain” (słabsza przesłanka)."""
    moved = p.ind.get("domain", frozenset()) & redirectors
    if not moved:
        return
    rest = p.ind["domain"] - moved
    if rest:
        p.ind["domain"] = rest
    else:
        del p.ind["domain"]
    p.ind["redirect_domain"] = p.ind.get("redirect_domain", frozenset()) | moved


# ---------------------------------------------------------------- kontekst porównań


class LinkContext:
    """Wspólny kontekst porównań: konfiguracja, statystyki korpusu, przygotowane materiały, indeks tekstu.

    - `features`, `by_id`, `ids` — materiały bazowe (kolejność = wiersze indeksu tekstu),
    - indeks TF-IDF n-gramów znakowych 3–5 (`char_wb`, `sublinear_tf`) — HashingVectorizer + TfidfTransformer
      (stała pamięć niezależnie od słownika), dopasowany na tekstach bazy; `tfidf_sim(a, b)`,
    - `add(fs)` dołącza nowy materiał (transformacja bez ponownego dopasowania — analiza nowego materiału),
    - `minhash(id)` — obiekt datasketch.MinHash odtworzony z FeatureSet.text.minhash.
    """

    def __init__(self, cfg: Settings, features: Iterable[FeatureSet], stats: CorpusStats | None = None):
        self.cfg = cfg
        by_id: dict[str, FeatureSet] = {}
        for fs in features:
            if fs is not None:
                by_id[fs.material_id] = fs
        self.by_id = by_id
        self.features = list(by_id.values())
        self.ids = list(by_id)
        self.stop = load_stoplists()
        self.prep: dict[str, Prepared] = {}
        for fs in self.features:
            self.prep[fs.material_id] = self._safe_prepare(fs)
        if stats is not None and stats.n > 0:
            self.stats = stats
        else:  # statystyki z już policzonych wskaźników (bez ponownej ekstrakcji)
            self.stats = stats_from_values({**p.ind, **continuous_values(p.fs, self.stop)} for p in self.prep.values())
        # domeny pośrednie znane w korpusie (skracacze, cloakery, TDS) — także dla materiałów z martwym łańcuchem
        self.redirectors: frozenset[str] = frozenset(
            {v for p in self.prep.values() for v in p.ind.get("redirect_domain", ())}
            | set(self.stats.df.get("redirect_domain", {}))
        )
        for p in self.prep.values():
            _reclassify(p, self.redirectors)
        self.extra: dict[str, Prepared] = {}
        self._extra_vec: dict[str, tuple] = {}
        self._extra_sims: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._pair_cache: dict[tuple[int, int], tuple[float, float]] = {}
        self._seg_cache: dict[tuple[int, int], tuple[float, int]] = {}
        self._extra_seg: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._minhash_objs: dict = {}
        self.row: dict[str, int] = {mid: i for i, mid in enumerate(self.ids)}
        self.X = self.Xc = None  # macierze TF-IDF (całość tekstu / treść reklamy), CSR float32
        self.Xi = self.Xo = None  # macierze TF-IDF kanałów treści reklamy: tekst z wejścia / OCR
        self.S = self.Sc = None  # gęste macierze podobieństw (dla małych zbiorów)
        self.Sseg = self.Sseg_arg = None  # najwyższy kosinus między kanałami (gęsto) i para kanałów (_SEG_COMBOS)
        self._hasher = self._tfidf = None
        self._fit_text()

    # ------------------------------------------------------------ przygotowanie
    def _safe_prepare(self, fs: FeatureSet) -> Prepared:
        try:
            p = prepare(fs, self.stop)
        except Exception as e:  # noqa: BLE001
            log.warning("Nie udało się przygotować materiału %s: %s", fs.material_id, e)
            p = Prepared(id=fs.material_id, fs=fs, ind={})
        redirectors = getattr(self, "redirectors", None)
        if redirectors:
            _reclassify(p, redirectors)
        return p

    def weight(self, kind: str) -> float:
        return weight(self.cfg.link.weights, kind)

    def rarity(self, kind: str, value: str) -> float:
        return rarity_factor(kind, value, self.stats, enabled=self.cfg.link.rarity)

    def prepared(self, fs: FeatureSet) -> Prepared:
        """Przygotowany materiał: z bazy (ten sam obiekt), dołączony przez add() albo przygotowany teraz."""
        p = self.prep.get(fs.material_id)
        if p is not None and p.fs is fs:
            return p
        p = self.extra.get(fs.material_id)
        if p is not None and p.fs is fs:
            return p
        return self.add(fs)

    def add(self, fs: FeatureSet) -> Prepared:
        """Dołącza materiał spoza bazy (np. nowy do analizy): przygotowanie + wektory tekstu (bez refit)."""
        p = self._safe_prepare(fs)
        self.extra.pop(fs.material_id, None)
        self.extra[fs.material_id] = p
        self._extra_vec.pop(fs.material_id, None)
        self._extra_sims.pop(fs.material_id, None)
        self._extra_seg.pop(fs.material_id, None)
        while len(self.extra) > MAX_EXTRA:  # długo działający UI: najstarsze dołączone materiały wypadają
            old_id = next(iter(self.extra))
            del self.extra[old_id]
            self._extra_vec.pop(old_id, None)
            self._extra_sims.pop(old_id, None)
            self._extra_seg.pop(old_id, None)
        if self._tfidf is not None:
            try:
                self._extra_vec[fs.material_id] = self._transform(fs)
            except Exception as e:  # noqa: BLE001
                log.warning("TF-IDF dla %s niedostępny: %s", fs.material_id, e)
        return p

    def is_base(self, p: Prepared) -> bool:
        return self.prep.get(p.id) is p

    # ------------------------------------------------------------ tekst
    def _fit_text(self) -> None:
        docs = [_doc(fs.text.text) for fs in self.features]
        nonempty = [i for i, d in enumerate(docs) if d]
        if not nonempty:
            return
        try:
            from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

            hasher = HashingVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                n_features=HASH_FEATURES,
                alternate_sign=False,
                norm=None,
                lowercase=False,
                dtype=np.float32,
            )
            counts = hasher.transform(docs)
            tfidf = TfidfTransformer(sublinear_tf=True).fit(counts[nonempty])
            X = tfidf.transform(counts).tocsr().astype(np.float32)
            Xc = tfidf.transform(hasher.transform([_creative_doc(fs) for fs in self.features])).tocsr()
            Xc = Xc.astype(np.float32)
            X.sum_duplicates()
            Xc.sum_duplicates()
            self._hasher, self._tfidf, self.X, self.Xc = hasher, tfidf, X, Xc
            if len(self.ids) <= DENSE_MAX:
                self.S = np.asarray((X @ X.T).toarray(), dtype=np.float32)
                self.Sc = np.asarray((Xc @ Xc.T).toarray(), dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            log.warning("Indeks TF-IDF niedostępny: %s", e)
            self._hasher = self._tfidf = self.X = self.Xc = self.S = self.Sc = None
            return
        try:  # kanały treści reklamy (wejście / OCR) — porównanie także na krzyż
            segs = [_segment_docs(fs) for fs in self.features]
            Xi = self._seg_matrix([s[0] for s in segs])
            Xo = self._seg_matrix([s[1] for s in segs])
            self.Xi, self.Xo = Xi, Xo
            if len(self.ids) <= DENSE_MAX:
                s_ii = np.asarray((Xi @ Xi.T).toarray(), dtype=np.float32)
                s_io = np.asarray((Xi @ Xo.T).toarray(), dtype=np.float32)
                s_oo = np.asarray((Xo @ Xo.T).toarray(), dtype=np.float32)
                stack = np.stack([s_ii, s_io, s_io.T, s_oo])
                self.Sseg = stack.max(axis=0)
                self.Sseg_arg = stack.argmax(axis=0).astype(np.int8)
        except Exception as e:  # noqa: BLE001
            log.warning("Indeks kanałów treści niedostępny: %s", e)
            self.Xi = self.Xo = self.Sseg = self.Sseg_arg = None

    def _seg_matrix(self, docs: list[str]):
        """Wektory kanałów treści: zbiory n-gramów znakowych 3–5 (binarnie, L2) — miara „niemal identycznego
        tekstu” odporna na literówki i błędy OCR. IDF celowo pominięty: unikalne n-gramy literówki miałyby
        najwyższą wagę i zaniżały podobieństwo kopii tej samej treści (wspólny szablon kampanii ma niski IDF)."""
        from sklearn.preprocessing import normalize

        counts = self._hasher.transform(docs).tocsr()
        counts.sum_duplicates()
        counts.data[:] = 1.0
        X = normalize(counts, norm="l2", copy=False).astype(np.float32)
        X.sum_duplicates()
        return X

    def _transform(self, fs: FeatureSet) -> tuple:
        def vec(doc: str):
            return self._tfidf.transform(self._hasher.transform([doc])).tocsr().astype(np.float32)

        seg_in, seg_ocr = _segment_docs(fs)
        return vec(_doc(fs.text.text)), vec(_creative_doc(fs)), self._seg_matrix([seg_in]), self._seg_matrix([seg_ocr])

    def text_vector(self, text: str):
        """Wektor TF-IDF dowolnego tekstu (np. w UI) albo None, gdy indeks niedostępny."""
        if self._tfidf is None:
            return None
        return self._tfidf.transform(self._hasher.transform([_doc(text)])).tocsr()

    def _vectors(self, p: Prepared) -> tuple | None:
        if self._tfidf is None:
            return None
        if self.is_base(p):
            i = self.row[p.id]
            if self.Xi is None:
                return self.X[i], self.Xc[i], None, None
            return self.X[i], self.Xc[i], self.Xi[i], self.Xo[i]
        vec = self._extra_vec.get(p.id) if self.extra.get(p.id) is p else None
        if vec is None:
            vec = self._transform(p.fs)
            if self.extra.get(p.id) is p:
                self._extra_vec[p.id] = vec
        return vec

    def base_text_sims(self, p: Prepared) -> tuple[np.ndarray, np.ndarray] | None:
        """Kosinusy TF-IDF materiału względem WSZYSTKICH materiałów bazy: (całość tekstu, treść reklamy)."""
        if self._tfidf is None or self.X is None:
            return None
        if self.is_base(p):
            i = self.row[p.id]
            if self.S is not None:
                return self.S[i], self.Sc[i]
            sj = np.asarray((self.X @ self.X[i].transpose().tocsc()).toarray()).ravel()
            sc = np.asarray((self.Xc @ self.Xc[i].transpose().tocsc()).toarray()).ravel()
            return sj, sc
        own = self.extra.get(p.id) is p
        hit = self._extra_sims.get(p.id) if own else None
        if hit is None:
            x, xc = self._vectors(p)[:2]
            hit = (
                np.asarray((self.X @ x.transpose().tocsc()).toarray(), dtype=np.float32).ravel(),
                np.asarray((self.Xc @ xc.transpose().tocsc()).toarray(), dtype=np.float32).ravel(),
            )
            if own:
                self._extra_sims[p.id] = hit
        return hit

    def text_sims(self, pa: Prepared, pb: Prepared) -> tuple[float, float]:
        """(kosinus TF-IDF całego tekstu, kosinus samej treści reklamy)."""
        if self._tfidf is None:
            return 0.0, 0.0
        base_a, base_b = self.is_base(pa), self.is_base(pb)
        if base_a and base_b:
            i, j = self.row[pa.id], self.row[pb.id]
            if self.S is not None:
                return float(self.S[i, j]), float(self.Sc[i, j])
            key = (i, j) if i < j else (j, i)
            hit = self._pair_cache.get(key)
            if hit is None:
                hit = (_rowdot(self.X, i, j), _rowdot(self.Xc, i, j))
                self._pair_cache[key] = hit
            return hit
        if base_a != base_b:  # materiał spoza bazy vs baza: wektor podobieństw liczony raz
            other, base = (pb, pa) if base_a else (pa, pb)
            if self.extra.get(other.id) is other:
                sims = self.base_text_sims(other)
                if sims is not None:
                    i = self.row[base.id]
                    return float(sims[0][i]), float(sims[1][i])
        va, vb = self._vectors(pa), self._vectors(pb)
        if va is None or vb is None:
            return 0.0, 0.0
        return float(va[0].multiply(vb[0]).sum()), float(va[1].multiply(vb[1]).sum())

    def segment_sims(self, pa: Prepared, pb: Prepared) -> tuple[float, int, tuple[int, int]]:
        """Najwyższy kosinus TF-IDF między kanałami treści reklamy obu materiałów (wejście–wejście,
        wejście–OCR, OCR–wejście, OCR–OCR), długość krótszego z dopasowanych kanałów (znaki) i para
        kanałów (0 = tekst z wejścia, 1 = OCR) w kolejności (pa, pb).

        Ta sama treść podana raz jako tekst, raz jako zrzut ekranu, albo zrzut bez tekstu wejściowego
        porównany z pełnym rekordem — złączenie kanałów zaniżałoby podobieństwo (materiały częściowe)."""
        if self._tfidf is None or self.Xi is None or not any(pa.seg_chars) or not any(pb.seg_chars):
            return 0.0, 0, (0, 0)
        base_a, base_b = self.is_base(pa), self.is_base(pb)
        c, k = 0.0, -1
        if base_a and base_b:
            i, j = self.row[pa.id], self.row[pb.id]
            if self.Sseg is not None:
                c, k = float(self.Sseg[i, j]), int(self.Sseg_arg[i, j])
            else:
                key = (i, j)
                hit = self._seg_cache.get(key)
                if hit is None:
                    mats = (self.Xi, self.Xo)
                    vals = [_rowdot2(mats[sa], i, mats[sb], j) for sa, sb in _SEG_COMBOS]
                    hit = (max(vals), int(np.argmax(vals)))
                    self._seg_cache[key] = hit
                c, k = hit
        elif base_a != base_b and self.extra.get((pb if base_a else pa).id) is (pb if base_a else pa):
            other, base = (pb, pa) if base_a else (pa, pb)
            vals, args = self._extra_segment_sims(other)
            i = self.row[base.id]
            c, k = float(vals[i]), int(args[i])
            if base_a:  # macierz liczona jako (nowy, baza) — dla pary (baza, nowy) kanały zamienione
                k = _SEG_COMBOS.index(_SEG_COMBOS[k][::-1])
        else:
            va, vb = self._vectors(pa), self._vectors(pb)
            if va is None or vb is None or va[2] is None or vb[2] is None:
                return 0.0, 0, (0, 0)
            vals = [float(va[2 + sa].multiply(vb[2 + sb]).sum()) for sa, sb in _SEG_COMBOS]
            k = int(np.argmax(vals))
            c = vals[k]
        if k < 0 or c <= 0.0:
            return 0.0, 0, (0, 0)
        sa, sb = _SEG_COMBOS[k]
        return max(0.0, min(1.0, c)), min(pa.seg_chars[sa], pb.seg_chars[sb]), (sa, sb)

    def _extra_segment_sims(self, p: Prepared) -> tuple[np.ndarray, np.ndarray]:
        """Kanały materiału spoza bazy vs wszystkie materiały bazy: (najwyższy kosinus, indeks pary kanałów)."""
        hit = self._extra_seg.get(p.id)
        if hit is None:
            vec = self._vectors(p)
            mine = (vec[2], vec[3])
            base = (self.Xi, self.Xo)
            rows = [
                np.asarray((base[sb] @ mine[sa].transpose().tocsc()).toarray(), dtype=np.float32).ravel()
                for sa, sb in _SEG_COMBOS
            ]
            stack = np.stack(rows)
            hit = (stack.max(axis=0), stack.argmax(axis=0))
            self._extra_seg[p.id] = hit
        return hit

    def tfidf_sim(self, id_a: str, id_b: str) -> float:
        """Podobieństwo treści dwóch materiałów (max z całości tekstu i treści reklamy), 0..1."""
        pa = self.prep.get(id_a) or self.extra.get(id_a)
        pb = self.prep.get(id_b) or self.extra.get(id_b)
        if pa is None or pb is None:
            return 0.0
        return max(self.text_sims(pa, pb))

    def prepare_pairs(self, pairs: Iterable[tuple[str, str]]) -> None:
        """Hurtowe policzenie podobieństw tekstu dla par kandydatów (duże zbiory, bez macierzy gęstej)."""
        if self._tfidf is None or self.S is not None:
            return
        X, Xc, cache = self.X, self.Xc, self._pair_cache
        for a, b in pairs:  # iloczyn skalarny wierszy CSR przez przecięcie indeksów (~40 µs na parę)
            i, j = self.row.get(a), self.row.get(b)
            if i is None or j is None or i == j:
                continue
            key = (i, j) if i < j else (j, i)
            if key not in cache:
                cache[key] = (_rowdot(X, i, j), _rowdot(Xc, i, j))

    # ------------------------------------------------------------ gęstość sąsiedztwa (cechy ciągłe)
    def _density_index(self) -> dict:
        """Dla każdego obrazu i szablonu DOM bazy: ilu INNYCH materiałów ma obraz / szablon w promieniu
        „silnego” podobieństwa (pHash Δ ≤ 6, simhash DOM Δ ≤ 6). Wspólny układ zrzutów ekranu platformy
        albo popularny motyw CMS mają gęste sąsiedztwo → niższa waga (rzadkość dla cech ciągłych)."""
        cached = getattr(self, "_density", None)
        if cached is not None:
            return cached
        img_keys, img_owner, img_hash, dom_owner, dom_hash = [], [], [], [], []
        for i, mid in enumerate(self.ids):
            p = self.prep[mid]
            for k, (ph, _, _, _) in enumerate(p.images):
                if ph is not None:
                    img_keys.append((mid, k))
                    img_owner.append(i)
                    img_hash.append(ph)
            if p.dom is not None:
                dom_owner.append(i)
                dom_hash.append(p.dom)
        dens: dict = {
            "img_owner": np.asarray(img_owner, dtype=np.int64),
            "img_hash": np.asarray(img_hash, dtype=np.uint64),
        }
        dens["dom_owner"] = np.asarray(dom_owner, dtype=np.int64)
        dens["dom_hash"] = np.asarray(dom_hash, dtype=np.uint64)
        counts = _neighbour_counts(dens["img_owner"], dens["img_hash"], PHASH_STRONG, len(self.ids))
        dens["img"] = {key: int(c) for key, c in zip(img_keys, counts.tolist(), strict=True)}
        counts = _neighbour_counts(dens["dom_owner"], dens["dom_hash"], DOM_MAX, len(self.ids))
        dens["dom"] = {self.ids[o]: int(c) for o, c in zip(dom_owner, counts.tolist(), strict=True)}
        self._density = dens
        return dens

    def neighbour_count(self, p: Prepared, what: str, k: int = 0) -> int:
        """Liczba innych materiałów bazy w promieniu silnego podobieństwa (obraz k / szablon DOM)."""
        dens = self._density_index()
        if self.is_base(p):
            return dens["img"].get((p.id, k), 0) if what == "img" else dens["dom"].get(p.id, 0)
        if what == "img":
            ph = p.images[k][0] if k < len(p.images) else None
            owner, hashes, radius = dens["img_owner"], dens["img_hash"], PHASH_STRONG
        else:
            ph, owner, hashes, radius = p.dom, dens["dom_owner"], dens["dom_hash"], DOM_MAX
        if ph is None or not len(hashes):
            return 0
        near = owner[np.bitwise_count(hashes ^ np.uint64(ph)) <= radius]
        own = self.row.get(p.id, -1)
        return int(len(set(near.tolist()) - {own}))

    def density_rarity(self, p: Prepared, what: str, k: int = 0) -> tuple[float, int]:
        """(współczynnik rzadkości, liczba materiałów w sąsiedztwie wraz z tym) dla obrazu / szablonu."""
        df = self.neighbour_count(p, what, k) + 1
        if not self.cfg.link.rarity:
            return 1.0, df
        return rarity_from_df(df, max(len(self.ids), 1), "image_density"), df

    # ------------------------------------------------------------ MinHash
    def minhash(self, mid: str):
        """datasketch.MinHash odtworzony z sygnatury (None, gdy brak sygnatury)."""
        p = self.prep.get(mid) or self.extra.get(mid)
        if p is None or p.minhash is None:
            return None
        if mid not in self._minhash_objs:
            from datasketch import MinHash

            perms = self._minhash_objs.get(("perms", len(p.minhash)))
            if perms is None:
                perms = MinHash(num_perm=len(p.minhash), scheme="legacy").permutations
                self._minhash_objs[("perms", len(p.minhash))] = perms
            self._minhash_objs[mid] = MinHash(
                num_perm=len(p.minhash), hashvalues=p.minhash, scheme="legacy", permutations=perms
            )
        return self._minhash_objs[mid]

    @staticmethod
    def jaccard(pa: Prepared, pb: Prepared) -> float | None:
        a, b = pa.minhash, pb.minhash
        if a is None or b is None or a.shape != b.shape:
            return None
        return float(np.count_nonzero(a == b)) / float(a.size)


def _neighbour_counts(owner: np.ndarray, hashes: np.ndarray, radius: int, n_owners: int) -> np.ndarray:
    """Dla każdego hasha: liczba RÓŻNYCH innych właścicieli z hashem w odległości Hamminga ≤ radius."""
    m = len(hashes)
    out = np.zeros(m, dtype=np.int64)
    if m < 2:
        return out
    block = max(1, min(512, 2_000_000 // max(n_owners, 1)))
    for s in range(0, m, block):
        e = min(m, s + block)
        near = np.bitwise_count(hashes[s:e, None] ^ hashes[None, :]) <= radius  # (B, M)
        hit = np.zeros((e - s, n_owners), dtype=bool)
        rows, cols = np.nonzero(near)
        hit[rows, owner[cols]] = True
        hit[np.arange(e - s), owner[s:e]] = False  # bez własnego materiału
        out[s:e] = hit.sum(axis=1)
    return out


def _rowdot(X, i: int, j: int) -> float:
    """Iloczyn skalarny wierszy i, j macierzy CSR o posortowanych indeksach (sum_duplicates)."""
    s0, e0 = X.indptr[i], X.indptr[i + 1]
    s1, e1 = X.indptr[j], X.indptr[j + 1]
    if s0 == e0 or s1 == e1:
        return 0.0
    if e0 - s0 > e1 - s1:  # szukamy krótszego wiersza w dłuższym
        s0, e0, s1, e1 = s1, e1, s0, e0
    ia, ib = X.indices[s0:e0], X.indices[s1:e1]
    pos = np.searchsorted(ib, ia)
    pos[pos >= len(ib)] = 0
    hit = ib[pos] == ia
    if not hit.any():
        return 0.0
    return float(np.dot(X.data[s0:e0][hit], X.data[s1:e1][pos[hit]]))


def _rowdot2(A, i: int, B, j: int) -> float:
    """Iloczyn skalarny wiersza i macierzy A i wiersza j macierzy B (CSR, posortowane indeksy)."""
    s0, e0 = A.indptr[i], A.indptr[i + 1]
    s1, e1 = B.indptr[j], B.indptr[j + 1]
    if s0 == e0 or s1 == e1:
        return 0.0
    ia, da = A.indices[s0:e0], A.data[s0:e0]
    ib, db = B.indices[s1:e1], B.data[s1:e1]
    if len(ia) > len(ib):  # szukamy krótszego wiersza w dłuższym
        ia, da, ib, db = ib, db, ia, da
    pos = np.searchsorted(ib, ia)
    pos[pos >= len(ib)] = 0
    hit = ib[pos] == ia
    if not hit.any():
        return 0.0
    return float(np.dot(da[hit], db[pos[hit]]))


# ---------------------------------------------------------------- ocena pary


@dataclass
class _Item:
    kind: str
    weight: float
    value: str | None = None
    similarity: float | None = None
    detail: str = ""
    rarity: float = 1.0
    df: int | None = None
    tier: str = ""  # strong | weak | soft (puste = wg rodzaju)

    def __post_init__(self) -> None:
        if not self.tier:
            if self.kind in SOFT_KINDS:
                self.tier = "soft"
            elif self.kind in STRONG_KINDS:
                self.tier = "strong"
            else:
                self.tier = "weak"

    @property
    def soft(self) -> bool:
        return self.tier == "soft"


@dataclass
class _Content:
    jaccard: float | None = None
    cos_text: float = 0.0
    cos_creative: float = 0.0
    cos_segment: float = 0.0  # najlepsza para kanałów treści reklamy (wejście / OCR, także na krzyż)
    image_dist: int | None = None
    image_dhash: int | None = None
    image_identical: bool = False


@dataclass
class PairScore:
    """Pełna ocena pary (także poniżej progu) — do wyjaśnień w UI („dlaczego (nie) powiązano”)."""

    a: str
    b: str
    score: float
    relation: Relation | None
    evidence: list[Evidence]
    coverage: float

    def to_link(self) -> Link | None:
        if self.relation is None:
            return None
        return Link(
            a=self.a, b=self.b, score=self.score, relation=self.relation, evidence=self.evidence, coverage=self.coverage
        )


_LABELS = {
    "tracker_id": "wspólny identyfikator {v}",
    "crypto_wallet": "ten sam portfel kryptowalut {v}",
    "phone": "ten sam numer telefonu {v}",
    "email": "ten sam adres e-mail {v}",
    "telegram": "to samo konto Telegram {v}",
    "whatsapp": "ten sam numer WhatsApp {v}",
    "iban": "ten sam numer rachunku (IBAN) {v}",
    "tls_fingerprint": "ten sam certyfikat TLS ({v})",
    "cert_domain": "domena {v} w tym samym certyfikacie TLS",
    "domain": "wspólna domena {v}",
    "redirect_domain": "wspólna domena pośrednicząca w przekierowaniu {v}",
    "host": "wspólny host {v}",
    "short_url": "ten sam skrócony link {v}",
    "script_domain": "skrypty ładowane z tej samej domeny {v}",
    "favicon": "ta sama ikona strony (favicon mmh3 {v})",
    "form_action": "formularz wysyła dane pod ten sam adres {v}",
    "form_fields": "ten sam nietypowy zestaw pól formularza ({v})",
    "form_offer": "ten sam identyfikator oferty w formularzu ({v})",
    "ip": "wspólny adres IP {v}",
    "asn": "ta sama sieć (ASN) {v}",
    "nameserver": "wspólny serwer DNS {v}",
    "title": "ten sam tytuł strony „{v}”",
    "person": "ta sama osoba ({v})",
    "brand": "ta sama podszywana marka ({v})",
    "narrative": "ta sama narracja ({v})",
}


def _fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def _indicator_detail(kind: str, value: str, fa: FeatureSet, fb: FeatureSet) -> str:
    shown = display_value(kind, value, fa, fb)
    if kind == "tls_fingerprint" and len(shown) > 16:
        shown = shown[:16] + "…"
    if kind == "title" and len(shown) > 80:
        shown = shown[:80] + "…"
    return _LABELS.get(kind, kind + " {v}").format(v=shown)


def _rarity_note(df: int | None, rarity: float) -> str:
    if rarity >= RARITY_NOTE_BELOW or not df:
        return ""
    return f" — wartość częsta ({df} mat. w zbiorze), waga obniżona"


def _indicator_items(pa: Prepared, pb: Prepared, ctx: LinkContext) -> list[_Item]:
    items: list[_Item] = []
    ia, ib = pa.ind, pb.ind
    shared_domains = ia.get("domain", frozenset()) & ib.get("domain", frozenset())
    for kind in INDICATOR_KINDS:
        if kind == "cert_domain":
            ca = ia.get("cert_domain", frozenset()) | ia.get("domain", frozenset())
            cb = ib.get("cert_domain", frozenset()) | ib.get("domain", frozenset())
            shared = {
                v
                for v in (ca & cb) - shared_domains
                if v in ia.get("cert_domain", frozenset()) or v in ib.get("cert_domain", frozenset())
            }
        else:
            va, vb = ia.get(kind), ib.get(kind)
            if not va or not vb:
                continue
            shared = va & vb
        if not shared:
            continue
        w = ctx.weight(kind)
        if w <= 0:
            continue
        scored = []
        for v in shared:
            if kind == "redirect_domain":  # pośrednik bywa też domeną końcową martwych materiałów
                df = (
                    document_frequency("redirect_domain", v, ctx.stats) + document_frequency("domain", v, ctx.stats) - 1
                )
                r = rarity_from_df(df, ctx.stats.n, kind) if ctx.cfg.link.rarity else 1.0
            else:
                r = ctx.rarity(kind, v)
                df = document_frequency(kind, v, ctx.stats)
            scored.append((w * r, v, r, df))
        scored.sort(key=lambda t: (-t[0], t[1]))
        if kind == "keyword":
            if len(scored) < 2:
                continue
            total = min(KEYWORD_CAP, sum(t[0] for t in scored))
            top = [t[1] for t in scored[:5]]
            more = f" (+{len(scored) - 5})" if len(scored) > 5 else ""
            items.append(
                _Item(
                    kind="keyword",
                    weight=total,
                    value=", ".join(top),
                    detail=f"wspólne słowa kluczowe: {', '.join(top)}{more}",
                )
            )
            continue
        for rank, (wt, v, r, df) in enumerate(scored[: MAX_VALUES.get(kind, 2)]):
            wt = wt if rank == 0 else 0.5 * wt
            detail = _indicator_detail(kind, v, pa.fs, pb.fs) + _rarity_note(df, r)
            items.append(_Item(kind=kind, weight=wt, value=v, detail=detail, rarity=r, df=df))
    # rejestrator + data rejestracji (tylko dla różnych domen — ta sama domena jest już przesłanką)
    if (
        not shared_domains
        and pa.registrar
        and pa.registrar == pb.registrar
        and pa.created is not None
        and pb.created is not None
    ):
        days = abs((pa.created - pb.created).total_seconds()) / 86400.0
        if days <= REG_DAYS:
            w = ctx.weight("registrar_created") * (1.0 if days <= 1.0 else 0.75)
            if pa.registrar_common:  # popularny rejestrator: zbieżność dat to co najwyżej słaba przesłanka
                w *= 0.5
            reg_name = pa.fs.infra.registrar or pa.registrar
            gap = max(1, round(days))
            when = "tego samego dnia" if days < 1.0 else f"w odstępie {gap} {'dnia' if gap == 1 else 'dni'}"
            items.append(
                _Item(
                    kind="registrar_created",
                    weight=w,
                    value=pa.registrar,
                    similarity=round(days, 2),
                    detail=f"domeny zarejestrowane u tego samego rejestratora ({reg_name}) {when}",
                )
            )
    return items


def _cos_to_jaccard(c: float) -> float:
    """Kosinus n-gramów znakowych ≥ 0.8 → równoważny Jaccard 0.5…1.0 (szum OCR psuje shingle słów, n-gramy nie)."""
    return 0.5 + 0.5 * (c - TFIDF_STRONG) / (1.0 - TFIDF_STRONG) if c >= TFIDF_STRONG else 0.0


def _text_items(pa: Prepared, pb: Prepared, ctx: LinkContext, content: _Content) -> list[_Item]:
    items: list[_Item] = []
    if pa.n_words >= MIN_MINHASH_WORDS and pb.n_words >= MIN_MINHASH_WORDS:
        content.jaccard = ctx.jaccard(pa, pb)
    text_ok = pa.n_chars >= MIN_TEXT_CHARS and pb.n_chars >= MIN_TEXT_CHARS
    creative_ok = pa.creative_chars >= MIN_TEXT_CHARS and pb.creative_chars >= MIN_TEXT_CHARS
    if text_ok or creative_ok:
        cj, cc = ctx.text_sims(pa, pb)
        content.cos_text = max(0.0, min(1.0, cj)) if text_ok else 0.0
        content.cos_creative = max(0.0, min(1.0, cc)) if creative_ok else 0.0
    c = max(content.cos_text, content.cos_creative)
    creative_wins = content.cos_creative >= content.cos_text
    what_c = "treść reklamy" if creative_wins else "treść"
    # krótkie teksty („Sprawdź teraz!”) bywają identyczne w niezależnych reklamach — waga rośnie z długością
    shorter = min(pa.creative_chars, pb.creative_chars) if creative_wins else min(pa.n_chars, pb.n_chars)
    shorter_mh = min(pa.n_chars, pb.n_chars)
    # kanały treści reklamy osobno (materiał częściowy: sam tekst / sam zrzut vs pełny rekord)
    channels = ""
    if creative_ok:
        seg_c, seg_len, (sa, sb) = ctx.segment_sims(pa, pb)
        content.cos_segment = seg_c
        if seg_c > c + 1e-6 and seg_len >= MIN_TEXT_CHARS:
            c, creative_wins, shorter, what_c = seg_c, True, seg_len, "treść reklamy"
            if sa != sb:
                channels = " — w jednym materiale tekst, w drugim napis na obrazie (OCR)"
            elif sa == 1:
                channels = " — napisy na obrazach (OCR)"
    # tekst niemal identyczny: Jaccard sygnatur MinHash albo bardzo wysoki kosinus n-gramów znakowych
    mh = tf = None
    j = content.jaccard if content.jaccard is not None else 0.0
    j_cos = _cos_to_jaccard(c)
    j_eff = max(j, j_cos)
    if j_eff >= MINHASH_MIN:
        length = shorter_mh if j >= j_cos else shorter
        lf = min(1.0, length / TEXT_FULL_CHARS)
        w = ctx.weight("text_minhash") * (j_eff - MINHASH_MIN) / (1.0 - MINHASH_MIN) * lf
        if j >= j_cos:
            what = "teksty niemal identyczne" if j >= 0.8 else "teksty w dużej części wspólne"
            detail = f"{what} (Jaccard {_fmt(j)})"
        else:
            detail = f"{what_c} niemal identyczna (cos {_fmt(c)}){channels}"
        if lf < 1.0:
            detail += f" — krótki tekst ({length} zn.), waga obniżona"
        tier = "strong" if length >= TEXT_STRONG_CHARS else "weak"
        # tekst ≥ TEXT_STRONG_CHARS może potwierdzać powiązanie (jak przy TF-IDF); waga nadal rośnie z długością
        mh = _Item(
            kind="text_minhash",
            weight=w,
            similarity=round(j_eff, 3),
            detail=detail,
            tier=tier,
            rarity=max(lf, RARE_ENOUGH) if length >= TEXT_STRONG_CHARS else lf,
        )
    if c >= TFIDF_MIN:
        lf = min(1.0, shorter / TEXT_FULL_CHARS)
        w = ctx.weight("text_tfidf") * ((c - TFIDF_MIN) / (1.0 - TFIDF_MIN)) ** TFIDF_GAMMA * lf
        strength = "bardzo podobna" if c >= 0.85 else "podobna"
        tf = _Item(
            kind="text_tfidf",
            weight=w,
            similarity=round(c, 3),
            detail=f"{strength} {what_c} (cos {_fmt(c)}){channels}",
            # potwierdza powiązanie tylko wyraźne podobieństwo (cos ≥ 0.7) dostatecznie długich tekstów;
            # teksty o tej samej narracji mają zwykle cos 0.4–0.6
            rarity=(max(lf, RARE_ENOUGH) if shorter >= TEXT_STRONG_CHARS else lf)
            if c >= TFIDF_CORROB
            else min(lf, 0.5),
        )
    if mh and tf:  # bez podwójnego liczenia: słabsza miara tekstu liczy się w połowie
        weaker = mh if mh.weight < tf.weight else tf
        weaker.weight *= 0.5
    items.extend(x for x in (mh, tf) if x is not None and x.weight > 0)
    return items


def _image_items(pa: Prepared, pb: Prepared, ctx: LinkContext, content: _Content) -> list[_Item]:
    items: list[_Item] = []
    best = None  # (dist, identical, index_a, index_b, dhash_dist)
    for ka, (pha, dha, shaa, _) in enumerate(pa.images):
        for kb, (phb, dhb, shab, _) in enumerate(pb.images):
            if shaa and shaa == shab:
                if pha is None:  # pusty / jednolity obraz (zaślepka, pusty zrzut) — ten sam plik nic nie mówi
                    continue
                cand = (0, True, ka, kb, 0)
            elif pha is not None and phb is not None:
                dd = (dha ^ dhb).bit_count() if (dha is not None and dhb is not None) else None
                cand = ((pha ^ phb).bit_count(), False, ka, kb, dd)
            else:
                continue
            if best is None or (cand[0], not cand[1]) < (best[0], not best[1]):
                best = cand
    if best is not None:
        dist, identical, ka, kb, dd = best
        content.image_dist = dist
        content.image_dhash = dd
        content.image_identical = identical
        if dist <= PHASH_MAX:
            ra, dfa = ctx.density_rarity(pa, "img", ka)
            rb, dfb = ctx.density_rarity(pb, "img", kb)
            r, dfs = min(ra, rb), [dfa, dfb]
            w = ctx.weight("image_phash") * (1.0 - dist / PHASH_SCALE) * r
            if not identical and dd is not None and dist > 4 and dd > DHASH_SUSPECT:
                w *= 0.5
            if identical:
                detail = "identyczny plik graficzny (SHA-256)"
            elif dist == 0:
                detail = "ta sama grafika (pHash Δ=0)"
            else:
                detail = f"ta sama lub przerobiona grafika (pHash Δ={dist})"
            detail += _rarity_note(max(dfs) if dfs else None, r)
            # podobny układ, ale zupełnie inny tekst reklamy → raczej wspólny szablon niż ta sama kreacja
            template_only = (
                not identical
                and min(pa.creative_chars, pb.creative_chars) >= TEXT_STRONG_CHARS
                and content.cos_creative < TEXT_CONTRADICTS
            )
            if template_only:
                w *= 0.5
                detail += " — ale inny tekst reklamy (wspólny szablon?), waga obniżona"
            items.append(
                _Item(
                    kind="image_phash",
                    weight=w,
                    similarity=round(1.0 - dist / 64.0, 3),
                    detail=detail,
                    rarity=r,
                    tier="strong" if (identical or (dist <= PHASH_STRONG and not template_only)) else "weak",
                )
            )
    if not items and pa.clip is not None and pb.clip is not None and pa.clip.shape[1] == pb.clip.shape[1]:
        cos = float(np.max(pa.clip @ pb.clip.T))
        if cos >= CLIP_MIN:
            w = ctx.weight("image_clip") * min(1.0, (cos - CLIP_MIN) / 0.1)
            items.append(
                _Item(
                    kind="image_clip",
                    weight=w,
                    similarity=round(cos, 3),
                    detail=f"podobna grafika (CLIP cos {_fmt(cos)})",
                )
            )
    return items


def _face_items(pa: Prepared, pb: Prepared, ctx: LinkContext) -> list[_Item]:
    if pa.faces is None or pb.faces is None or pa.faces.shape[1] != pb.faces.shape[1]:
        return []
    cos = float(np.max(pa.faces @ pb.faces.T))
    thr = float(ctx.cfg.faces.match_threshold)
    if cos < thr:
        return []
    span = max(1e-6, FACE_FULL - thr)
    w = ctx.weight("face") * (0.5 + 0.5 * min(1.0, max(0.0, (cos - thr) / span)))
    return [_Item(kind="face", weight=w, similarity=round(cos, 3), detail=f"ta sama twarz (cos {_fmt(cos)})")]


def _dom_items(pa: Prepared, pb: Prepared, ctx: LinkContext) -> list[_Item]:
    if pa.generic or pb.generic:
        return []
    identical_html = bool(pa.html_sha) and pa.html_sha == pb.html_sha
    if identical_html:
        dist, detail = 0, "identyczny kod HTML strony"
    elif pa.dom is not None and pb.dom is not None:
        dist = (pa.dom ^ pb.dom).bit_count()
        detail = f"ten sam szablon strony (simhash DOM Δ={dist})"
    else:
        return []
    if dist > DOM_MAX:
        return []
    ra, dfa = ctx.density_rarity(pa, "dom") if pa.dom is not None else (1.0, 1)
    rb, dfb = ctx.density_rarity(pb, "dom") if pb.dom is not None else (1.0, 1)
    r, dfs = min(ra, rb), [dfa, dfb]
    w = ctx.weight("dom_simhash") * (1.0 - dist / DOM_SCALE) * r
    detail += _rarity_note(max(dfs) if dfs else None, r)
    tier = "strong" if identical_html else "weak"
    return [
        _Item(kind="dom_simhash", weight=w, similarity=round(1.0 - dist / 64.0, 3), detail=detail, rarity=r, tier=tier)
    ]


def _apply_caps(items: list[_Item], related: float) -> None:
    """Limity: osoby ≤ 0.75 × próg; przesłanki miękkie ≤ min(0.875 × próg, max(0.5 × próg, 2 × twarde));
    bez przesłanki silnej i przy < 2 rodzajach przesłanek średnich — wynik < próg (brak powiązania)."""
    persons = [it for it in items if it.kind in ("person", "face")]
    total_p = sum(it.weight for it in persons)
    cap_p = PERSON_CAP * related
    if total_p > cap_p > 0:
        for it in persons:
            it.weight *= cap_p / total_p
    hard = sum(it.weight for it in items if not it.soft)
    soft_items = [it for it in items if it.soft]
    soft = sum(it.weight for it in soft_items)
    limit = min(SOFT_CAP * related, max(SOFT_FLOOR * related, SOFT_HARD_RATIO * hard))
    if soft > limit:
        f = limit / soft if soft > 0 else 0.0
        for it in soft_items:
            it.weight *= f
    # potwierdzenie: powiązanie wymaga przesłanki silnej albo ≥ 2 różnych rodzajów przesłanek średnich —
    # liczą się tylko wartości nieczęste w zbiorze (wspólna favicona hostingu + wspólny skracacz to wciąż „tło”)
    strong = any(it.tier == "strong" and it.rarity >= RARE_ENOUGH for it in items)
    weak_kinds = {it.kind for it in items if it.tier == "weak" and it.rarity >= RARE_ENOUGH}
    if not strong and len(weak_kinds) < 2:
        total = sum(it.weight for it in items)
        cap = CORROBORATION_CAP * related
        if total > cap > 0:
            for it in items:
                it.weight *= cap / total


def _relation(score: float, pa: Prepared, pb: Prepared, content: _Content, cfg: Settings) -> Relation | None:
    """Relacja pary: duplikat = ta sama treść (każda grupa treści obecna w obu materiałach niemal identyczna);
    sam wysoki wynik (wspólna infrastruktura) oznacza wariant tej samej kampanii, nie duplikat."""
    related, variant, _ = thresholds(cfg)
    if score < related:
        return None
    identical: list[bool] = []
    if pa.n_chars >= MIN_TEXT_CHARS and pb.n_chars >= MIN_TEXT_CHARS:
        j = content.jaccard or 0.0
        c = max(content.cos_text, content.cos_creative)
        identical.append(j >= TEXT_DUP_JACCARD or c >= TEXT_DUP_COSINE)
    if pa.images and pb.images and content.image_dist is not None:  # obrazy porównywalne (hash / SHA-256)
        d, dd = content.image_dist, content.image_dhash
        identical.append(content.image_identical or (d <= IMAGE_DUP_DIST and (dd is None or dd <= IMAGE_DUP_DHASH)))
    if identical:
        if all(identical):
            return Relation.DUPLICATE
    elif not pa.has_content and not pb.has_content and pa.urls & pb.urls:
        return Relation.DUPLICATE  # same adresy (bez treści) — ten sam materiał zgłoszony ponownie
    if score >= variant:
        return Relation.VARIANT
    return Relation.RELATED


def _coverage(pa: Prepared, pb: Prepared) -> float:
    union = pa.available | pb.available
    return round(len(pa.available & pb.available) / len(union), 3) if union else 0.0


def _evaluate(pa: Prepared, pb: Prepared, ctx: LinkContext) -> tuple[list[_Item], _Content]:
    """Przesłanki pary (po limitach, posortowane malejąco wg wkładu) + informacje o treści."""
    content = _Content()
    items: list[_Item] = []
    for part, fn in (
        ("wskaźniki", lambda: _indicator_items(pa, pb, ctx)),
        ("tekst", lambda: _text_items(pa, pb, ctx, content)),
        ("obraz", lambda: _image_items(pa, pb, ctx, content)),
        ("twarze", lambda: _face_items(pa, pb, ctx)),
        ("szablon", lambda: _dom_items(pa, pb, ctx)),
    ):
        try:
            items.extend(fn())
        except Exception as e:  # noqa: BLE001 — jedna grupa cech nie psuje porównania
            log.warning("Porównanie %s–%s (%s): %s", pa.id, pb.id, part, e)
    related, _, _ = thresholds(ctx.cfg)
    _apply_caps(items, related)
    items = [it for it in items if round(it.weight, 3) > 0]
    items.sort(key=lambda it: (-it.weight, it.kind, it.value or ""))
    return items, content


def _score(items: list[_Item]) -> float:
    return round(sum(round(it.weight, 3) for it in items), 3)


def _evidence(items: list[_Item]) -> list[Evidence]:
    return [
        Evidence(
            kind=it.kind,
            group=KIND_GROUP.get(it.kind, "infrastruktura"),
            detail=it.detail,
            value=it.value,
            similarity=it.similarity,
            weight=round(it.weight, 3),
        )
        for it in items
    ]


def score_pair(fa: FeatureSet, fb: FeatureSet, ctx: LinkContext) -> PairScore:
    """Pełna ocena pary (przesłanki z wkładami, wynik, relacja albo None poniżej progu) — np. do UI."""
    pa, pb = ctx.prepared(fa), ctx.prepared(fb)
    items, content = _evaluate(pa, pb, ctx)
    score = _score(items)
    return PairScore(
        a=fa.material_id,
        b=fb.material_id,
        score=score,
        relation=_relation(score, pa, pb, content, ctx.cfg),
        evidence=_evidence(items),
        coverage=_coverage(pa, pb),
    )


def compare_prepared(pa: Prepared, pb: Prepared, ctx: LinkContext) -> Link | None:
    if pa.id == pb.id:
        return None
    items, content = _evaluate(pa, pb, ctx)
    score = _score(items)
    relation = _relation(score, pa, pb, content, ctx.cfg)
    if relation is None:
        return None
    return Link(a=pa.id, b=pb.id, score=score, relation=relation, evidence=_evidence(items), coverage=_coverage(pa, pb))


def compare(fa: FeatureSet, fb: FeatureSet, ctx: LinkContext) -> Link | None:
    """Powiązanie dwóch materiałów z przesłankami albo None (wynik poniżej progu „powiązany”)."""
    if fa.material_id == fb.material_id:
        return None
    return compare_prepared(ctx.prepared(fa), ctx.prepared(fb), ctx)


__all__ = [
    "LinkContext",
    "PairScore",
    "Prepared",
    "compare",
    "compare_prepared",
    "prepare",
    "rarity_from_df",
    "score_pair",
    "thresholds",
]
