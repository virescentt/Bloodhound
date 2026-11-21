"""Ewaluacja: metryki grupowania i przypisań, gdy znane są etykiety kampanii (zbiór syntetyczny / NASK).

metrics.pairwise_prf / bcubed / ari_nmi — metryki na słownikach id → kampania / etykieta
run.evaluate(ws, cfg) -> dict              — pełny raport z katalogu roboczego
run.format_report(report) -> str           — raport po polsku
CLI: ``python -m slad.eval.run [katalog_roboczy] [--json raport.json]``
"""

from typing import Any

from slad.eval.metrics import ari_nmi, bcubed, pairwise_prf

__all__ = ["ari_nmi", "bcubed", "evaluate", "format_report", "gold_labels", "pairwise_prf"]


def __getattr__(name: str) -> Any:  # leniwy import run.py (czyste `python -m slad.eval.run`)
    if name in ("evaluate", "format_report", "gold_labels"):
        from slad.eval import run

        return getattr(run, name)
    raise AttributeError(f"module 'slad.eval' has no attribute {name!r}")
