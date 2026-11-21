"""Ekstrakcja cech: Material → FeatureSet (obraz → infrastruktura → tekst).

Kolejność ma znaczenie: OCR obrazów i tekst strony docelowej (z infrastruktury) zasilają cechy tekstowe.
Błąd jednego kroku nie przerywa pozostałych — trafia do FeatureSet.errors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slad.models import FeatureSet, Material
from slad.settings import ROOT, Settings

log = logging.getLogger(__name__)


@dataclass
class ExtractContext:
    """Wspólny kontekst ekstrakcji: konfiguracja, pamięć podręczna i ciężkie silniki (ładowane raz)."""

    settings: Settings
    cache_root: Path
    llm: Any = None  # slad.features.llm.LLMClient | None
    faces: Any = None  # slad.features.faces.FaceEngine | None
    clip: Any = None  # slad.features.clip.ClipEmbedder | None
    _caches: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(
        cls,
        cfg: Settings,
        workspace: Path | None = None,
        *,
        with_llm: bool = True,
        with_faces: bool = True,
    ) -> ExtractContext:
        cache_root = Path(workspace) / "cache" if workspace else ROOT / ".cache"
        ctx = cls(settings=cfg, cache_root=cache_root)
        if with_llm and cfg.llm.enabled:
            try:
                from slad.features.llm import LLMClient

                ctx.llm = LLMClient.from_settings(cfg.llm, cache_root / "llm")
            except Exception as e:  # noqa: BLE001 — LLM jest opcjonalny
                log.warning("LLM niedostępny, używam heurystyk: %s", e)
        if with_faces and cfg.faces.enabled:
            try:
                from slad.features.faces import FaceEngine

                ctx.faces = FaceEngine.from_settings(cfg.faces)
            except Exception as e:  # noqa: BLE001
                log.warning("Rozpoznawanie twarzy niedostępne: %s", e)
        if cfg.clip.enabled:
            try:
                from slad.features.clip import ClipEmbedder

                ctx.clip = ClipEmbedder.from_settings(cfg.clip)
            except Exception as e:  # noqa: BLE001
                log.warning("CLIP niedostępny: %s", e)
        return ctx

    def cache(self, name: str) -> Any:
        """JsonCache dla danej przestrzeni nazw (infra, ocr, favicon, dns, rdap, tls, …)."""
        if name not in self._caches:
            from slad.features.infra.cache import JsonCache

            self._caches[name] = JsonCache(self.cache_root / name, ttl_h=self.settings.network.cache_ttl_h)
        return self._caches[name]


def extract(material: Material, ctx: ExtractContext) -> FeatureSet:
    """Pełna ekstrakcja cech jednego materiału. Nigdy nie rzuca wyjątku z powodu jednej grupy cech."""
    fs = FeatureSet(material_id=material.id)

    # 1. Obrazy: hashe percepcyjne, kolory, twarze (+ OCR do cech tekstowych)
    ocr_texts: list[str] = []
    for p in material.image_paths:
        path = Path(p)
        try:
            from slad.features.visual import image_features

            fs.images.append(image_features(path, ctx))
        except Exception as e:  # noqa: BLE001
            fs.errors.append(f"obraz {path.name}: {e}")
        if ctx.settings.ocr.enabled:
            try:
                from slad.features.ocr import ocr_image

                text = ocr_image(path, ctx)
                if text:
                    ocr_texts.append(text)
            except Exception as e:  # noqa: BLE001
                fs.errors.append(f"OCR {path.name}: {e}")

    # 2. Infrastruktura (+ widoczny tekst strony docelowej)
    page_text = ""
    try:
        from slad.features.infra import infra_features

        fs.infra, page_text = infra_features(material, ctx)
    except Exception as e:  # noqa: BLE001
        fs.errors.append(f"infrastruktura: {e}")

    # 3. Tekst: wejście + OCR + strona → osoby, marki, narracja, IOC, MinHash
    try:
        from slad.features.text import text_features

        fs.text = text_features(material, ocr_texts, page_text, ctx)
    except Exception as e:  # noqa: BLE001
        fs.errors.append(f"tekst: {e}")

    fs.available = available_groups(fs)
    return fs


def available_groups(fs: FeatureSet) -> list[str]:
    groups: list[str] = []
    if fs.text.text.strip():
        groups.append("tekst")
    if fs.images:
        groups.append("obraz")
    if any(img.faces for img in fs.images):
        groups.append("twarze")
    inf = fs.infra
    if inf.domains or inf.ips or inf.tracker_ids or inf.dom_simhash or inf.favicon_mmh3 is not None:
        groups.append("infrastruktura")
    return groups
