"""Statystyki korpusu: w ilu materiałach występuje wartość wskaźnika → ważenie rzadkością (IDF).

Wspólny ASN Cloudflare, popularny serwer DNS czy narracja obecna w połowie zbioru nie mogą wiązać
niezależnych materiałów — ich waga maleje. Wartość wspólna dla jednej (nawet dużej) kampanii nie powinna być
karana, więc próg „wartości częstej” zależy od rodzaju wskaźnika:
- identyfikatory operatora (piksel, portfel, telefon, e-mail, Telegram, IBAN, certyfikat, skrócony link) —
  bez kary do 35% zbioru: ich częstość odzwierciedla wielkość kampanii, nie „tło internetu”,
- domena docelowa — do 25% zbioru,
- pozostałe (IP, ASN, DNS, favicona, tytuł, osoba, narracja, słowa kluczowe, obraz, szablon) — do 10% zbioru
  (najwyżej 50 materiałów).

CorpusStats.df przechowuje tylko wartości obecne w ≥ 2 materiałach (wartość jednokrotna i tak ma
współczynnik 1.0) — plik stats.json pozostaje mały także dla dziesiątek tysięcy materiałów. Wyjątek:
„redirect_domain” (wszystkie domeny pośrednie) — to zbiór znanych pośredników przekierowań korpusu.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from collections.abc import Iterable

from slad.link.indicators import Stoplists, indicators, load_stoplists, page_is_generic
from slad.models import CorpusStats, FeatureSet

log = logging.getLogger(__name__)

FREE_FRACTION = 0.10  # wartości obecne w ≤ 10% materiałów (≤ FREE_MAX) — pełna waga
FREE_MAX = 50
OPERATOR_KINDS = frozenset(
    {"tracker_id", "crypto_wallet", "phone", "email", "telegram", "whatsapp", "iban", "tls_fingerprint", "short_url"}
)
OPERATOR_FREE_FRACTION = 0.35
SITE_KINDS = frozenset({"domain", "image_density"})  # image_density: sąsiedztwo pHash / simhash DOM
SITE_FREE_FRACTION = 0.25
EXPONENT = 2.0  # stromość spadku wagi powyżej progu
FLOOR = 0.05  # minimalny współczynnik


def hex64(value: str | None) -> str | None:
    """Znormalizowany 64-bitowy hash hex (pHash, simhash) albo None."""
    if not value:
        return None
    v = str(value).strip().lower().removeprefix("0x")
    if not v or len(v) > 16:
        return None
    try:
        int(v, 16)
    except ValueError:
        return None
    return v.zfill(16)


def continuous_values(fs: FeatureSet, stop: Stoplists | None = None) -> dict[str, set[str]]:
    """Dokładne wartości cech ciągłych (do częstości): pHash obrazów i simhash szablonu strony."""
    out: dict[str, set[str]] = {}
    ph = {h for h in (hex64(img.phash) for img in fs.images) if h}
    if ph:
        out["image_phash"] = ph
    dom = hex64(fs.infra.dom_simhash)
    if dom and not page_is_generic(fs, stop):
        out["dom_simhash"] = {dom}
    return out


def corpus_stats(features: Iterable[FeatureSet]) -> CorpusStats:
    """Liczba materiałów z daną wartością wskaźnika (kind → wartość → df), tylko df ≥ 2."""
    stop = load_stoplists()
    seen: set[str] = set()

    def values():
        for fs in features:
            if fs is None or fs.material_id in seen:
                continue
            seen.add(fs.material_id)
            try:
                vals = indicators(fs, stop)
                vals.update(continuous_values(fs, stop))
            except Exception as e:  # noqa: BLE001 — jeden zły materiał nie psuje statystyk
                log.warning("Statystyki: pomijam materiał %s (%s)", fs.material_id, e)
                vals = {}
            yield vals

    return stats_from_values(values())


def stats_from_values(per_material: Iterable[dict]) -> CorpusStats:
    """CorpusStats z gotowych wskaźników materiałów (po jednym słowniku kind → wartości na materiał)."""
    counts: dict[str, Counter] = defaultdict(Counter)
    n = 0
    for values in per_material:
        n += 1
        for kind, vals in values.items():
            counts[kind].update(set(vals))
    df = {}
    for kind in sorted(counts):
        # domeny pośrednie (przekierowania) zachowujemy wszystkie: zbiór znanych pośredników całego korpusu
        min_df = 1 if kind == "redirect_domain" else 2
        kept = {v: c for v, c in sorted(counts[kind].items()) if c >= min_df}
        if kept:
            df[kind] = kept
    return CorpusStats(n=n, df=df)


def free_df(kind: str | None, n: int) -> float:
    """Liczba materiałów, do której wartość danego rodzaju nie jest jeszcze „częsta”."""
    if kind in OPERATOR_KINDS:
        return max(2.0, OPERATOR_FREE_FRACTION * n)
    if kind in SITE_KINDS:
        return max(2.0, SITE_FREE_FRACTION * n)
    return min(max(2.0, FREE_FRACTION * n), FREE_MAX)


def rarity_from_df(df: int | None, n: int, kind: str | None = None) -> float:
    """Współczynnik rzadkości (0, 1] dla wartości rodzaju `kind` obecnej w `df` z `n` materiałów."""
    if not df or n <= 0 or df <= 2:
        return 1.0
    free = free_df(kind, n)
    if df <= free:
        return 1.0
    r = math.log1p(n / df) / math.log1p(n / free)
    return float(min(1.0, max(FLOOR, r**EXPONENT)))


def document_frequency(kind: str, value: str, stats: CorpusStats | None) -> int:
    if stats is None:
        return 1
    return int(stats.df.get(kind, {}).get(value, 1))


def rarity_factor(kind: str, value: str, stats: CorpusStats | None, enabled: bool = True) -> float:
    """1.0 dla wartości rzadkich (1–2 materiały lub poniżej progu rodzaju), maleje dla częstych (min 0.05).

    Przykład (n = 100, np. ASN): df ≤ 10 → 1.0 · df = 20 → 0.56 · df = 40 → 0.27 · df = 70 → 0.14;
    identyfikator operatora (np. piksel): df ≤ 35 → 1.0 · df = 70 → 0.43.
    `enabled=False` (config: link.rarity: false) → zawsze 1.0.
    """
    if not enabled or stats is None:
        return 1.0
    return rarity_from_df(document_frequency(kind, value, stats), stats.n, kind)
