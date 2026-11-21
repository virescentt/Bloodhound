"""Wejście: dowolny zbiór (CSV/JSON/JSONL/XLSX, katalog obrazów/HTML, listy URL) → list[Material].

load_dataset(path)          — zbiór w dowolnym obsługiwanym formacie (szczegóły: loaders.py)
material_from_input(...)    — pojedynczy nowy materiał z formularza UI / CLI
synthetic.generate(out_dir) — syntetyczny zbiór z etykietami kampanii (testy i demo bez danych NASK)
"""

from slad.ingest.loaders import load_dataset, map_column, material_from_input, normalize_url, parse_date

__all__ = ["load_dataset", "map_column", "material_from_input", "normalize_url", "parse_date"]
