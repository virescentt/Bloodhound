"""Metryki jakości grupowania materiałów w kampanie (gdy znane są etykiety wzorcowe).

Wejście każdej metryki:
    pred: id materiału → identyfikator przewidzianej kampanii albo None (materiał niepogrupowany = osobny singleton)
    gold: id materiału → etykieta wzorcowa; materiały z etykietą None są pomijane

Liczone wyłącznie na materiałach z etykietą wzorcową (materiał bez wpisu w ``pred`` = singleton).
Konwencja dla przypadków brzegowych: brak par przewidzianych → precyzja 1,0 (nic nie połączono błędnie),
brak par wzorcowych → pełność 1,0; pusty zbiór → same zera.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping
from typing import Any

Pred = Mapping[str, Hashable | None]
Gold = Mapping[str, Any]


def _cells(pred: Pred, gold: Gold) -> tuple[Counter, Counter, Counter, int]:
    """Tablica kontyngencji: (klaster, etykieta) → liczba materiałów, rozmiary klastrów i etykiet."""
    cells: Counter = Counter()
    csize: Counter = Counter()
    lsize: Counter = Counter()
    n = 0
    for mid, label in gold.items():
        if label is None:
            continue
        c = pred.get(mid)
        ck = ("c", c) if c is not None else ("s", mid)  # niepogrupowany = własny klaster
        cells[(ck, label)] += 1
        csize[ck] += 1
        lsize[label] += 1
        n += 1
    return cells, csize, lsize, n


def _pairs(k: int) -> int:
    return k * (k - 1) // 2


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def pairwise_prf(pred: Pred, gold: Gold) -> dict[str, float]:
    """Precyzja / pełność / F1 na parach: para jest „przewidziana”, gdy oba materiały są w tej samej kampanii."""
    cells, csize, lsize, n = _cells(pred, gold)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = sum(_pairs(v) for v in cells.values())
    pp = sum(_pairs(v) for v in csize.values())
    gp = sum(_pairs(v) for v in lsize.values())
    precision = tp / pp if pp else 1.0
    recall = tp / gp if gp else 1.0
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}


def bcubed(pred: Pred, gold: Gold) -> dict[str, float]:
    """B-Cubed (Bagga & Baldwin): średnia po materiałach udziału „swoich” w klastrze i w etykiecie."""
    cells, csize, lsize, n = _cells(pred, gold)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = sum(v * v / csize[ck] for (ck, _), v in cells.items()) / n
    recall = sum(v * v / lsize[label] for (_, label), v in cells.items()) / n
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}


def ari_nmi(pred: Pred, gold: Gold) -> dict[str, float]:
    """Adjusted Rand Index i Normalized Mutual Information (scikit-learn)."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    items = [mid for mid, label in gold.items() if label is not None]
    if not items:
        return {"ari": 0.0, "nmi": 0.0}
    gold_ids: dict[Any, int] = {}
    pred_ids: dict[Any, int] = {}
    y_true = [gold_ids.setdefault(gold[mid], len(gold_ids)) for mid in items]
    y_pred = []
    for mid in items:
        c = pred.get(mid)
        key = ("c", c) if c is not None else ("s", mid)
        y_pred.append(pred_ids.setdefault(key, len(pred_ids)))
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
    }
