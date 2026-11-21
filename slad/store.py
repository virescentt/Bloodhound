"""Katalog roboczy (workspace): materiały, cechy, powiązania, kampanie — w prostych plikach JSON/JSONL.

Struktura opisana w docs/ARCHITEKTURA.md. Zapisy są atomowe (plik tymczasowy + rename).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path

from slad.models import Campaign, CorpusStats, FeatureSet, Link, Material

_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_name(mid: str) -> str:
    safe = _SAFE.sub("_", mid)[:80]
    if safe != mid:
        safe += "-" + hashlib.sha1(mid.encode()).hexdigest()[:8]
    return safe


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "features").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._materials: dict[str, Material] | None = None

    # ------------------------------------------------------------ ścieżki
    def path(self, name: str) -> Path:
        return self.root / name

    def cache_dir(self, name: str) -> Path:
        d = self.root / "cache" / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------ materiały
    def _load_materials(self) -> dict[str, Material]:
        if self._materials is None:
            items: dict[str, Material] = {}
            f = self.path("materials.jsonl")
            if f.exists():
                for line in f.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        m = Material.model_validate_json(line)
                        items[m.id] = m
            self._materials = items
        return self._materials

    def materials(self) -> list[Material]:
        return list(self._load_materials().values())

    def material(self, mid: str) -> Material | None:
        return self._load_materials().get(mid)

    def save_materials(self, items: Iterable[Material]) -> None:
        with self._lock:
            self._materials = {m.id: m for m in items}
            _atomic_write(
                self.path("materials.jsonl"),
                "".join(m.model_dump_json() + "\n" for m in self._materials.values()),
            )

    def add_material(self, m: Material) -> None:
        items = self._load_materials()
        items[m.id] = m
        self.save_materials(list(items.values()))

    # ------------------------------------------------------------ cechy
    def _feature_path(self, mid: str) -> Path:
        return self.root / "features" / f"{_safe_name(mid)}.json"

    def features(self, mid: str) -> FeatureSet | None:
        f = self._feature_path(mid)
        return FeatureSet.model_validate_json(f.read_text(encoding="utf-8")) if f.exists() else None

    def save_features(self, fs: FeatureSet) -> None:
        _atomic_write(self._feature_path(fs.material_id), fs.model_dump_json())

    def all_features(self) -> list[FeatureSet]:
        out = []
        for m in self.materials():
            fs = self.features(m.id)
            if fs is not None:
                out.append(fs)
        return out

    # ------------------------------------------------------------ powiązania, kampanie, statystyki
    def links(self) -> list[Link]:
        f = self.path("links.jsonl")
        if not f.exists():
            return []
        return [Link.model_validate_json(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]

    def save_links(self, links: Iterable[Link]) -> None:
        _atomic_write(self.path("links.jsonl"), "".join(lk.model_dump_json() + "\n" for lk in links))

    def campaigns(self) -> list[Campaign]:
        f = self.path("campaigns.json")
        if not f.exists():
            return []
        return [Campaign.model_validate(c) for c in json.loads(f.read_text(encoding="utf-8"))]

    def save_campaigns(self, campaigns: Iterable[Campaign]) -> None:
        _atomic_write(
            self.path("campaigns.json"),
            json.dumps([c.model_dump(mode="json") for c in campaigns], ensure_ascii=False, indent=1),
        )

    def campaign_of(self, mid: str) -> str | None:
        for c in self.campaigns():
            if mid in c.members:
                return c.id
        return None

    def stats(self) -> CorpusStats:
        f = self.path("stats.json")
        return CorpusStats.model_validate_json(f.read_text(encoding="utf-8")) if f.exists() else CorpusStats()

    def save_stats(self, stats: CorpusStats) -> None:
        _atomic_write(self.path("stats.json"), stats.model_dump_json())
