"""Interfejs graficzny Śladu Fejka (Streamlit).

Uruchomienie:  streamlit run slad/ui/app.py   (albo: python -m slad ui)
Czyste funkcje pomocnicze (modele → tabele, kolory, graf) są w ``components`` i ``graph`` —
testowalne bez uruchomionego Streamlit.
"""

from __future__ import annotations

from pathlib import Path

APP_PATH = Path(__file__).with_name("app.py")


def app_path() -> Path:
    """Ścieżka skryptu aplikacji — dla CLI: ``streamlit run <app_path()>``."""
    return APP_PATH
