"""Opcjonalny embedding semantyczny obrazu: CLIP (open_clip + torch) — „podobne wizualnie” kreacje.

Włączenie: `clip.enabled: true` w config/settings.yaml + pakiety open_clip_torch i torch
(requirements-extra.txt). Bez nich `ClipEmbedder.from_settings` zwraca None, a potok działa dalej
bez cechy `image_clip` (hashe percepcyjne i twarze pozostają).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from slad.settings import ClipSettings

log = logging.getLogger(__name__)

INSTALL_HINT = "zainstaluj: pip install -r requirements-extra.txt (open_clip_torch, torch)"


def _to_pil(image: Any):
    """PIL.Image (dowolny tryb) albo np.ndarray w konwencji PIL (RGB / RGBA / skala szarości) → PIL RGB."""
    from PIL import Image

    from slad.features.visual import to_rgb

    if isinstance(image, Image.Image):
        return to_rgb(image)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(np.nan_to_num(arr), 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return to_rgb(Image.fromarray(arr))


class ClipEmbedder:
    """Koder obrazów CLIP. `model` musi mieć `encode_image`, `preprocess` — transformację PIL → tensor."""

    def __init__(self, model: Any, preprocess: Any, *, device: str = "cpu", name: str = "") -> None:
        self._model = model
        self._preprocess = preprocess
        self.device = device
        self.name = name
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, cfg: ClipSettings) -> ClipEmbedder | None:
        """Embedder według konfiguracji; None, gdy wyłączony, brak pakietów albo model się nie wczytał."""
        if not cfg.enabled:
            return None
        try:
            import open_clip  # noqa: F401
            import torch  # noqa: F401
        except ImportError as e:
            log.warning("CLIP niedostępny — brak pakietu %s; %s", e.name or e, INSTALL_HINT)
            return None
        try:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(cfg.model, pretrained=cfg.pretrained)
            model.eval()
        except Exception as e:  # noqa: BLE001 — np. brak wag offline
            log.warning("Nie udało się wczytać modelu CLIP %s (%s): %s", cfg.model, cfg.pretrained, e)
            return None
        log.info("CLIP: %s (%s)", cfg.model, cfg.pretrained)
        return cls(model, preprocess, device="cpu", name=f"{cfg.model}/{cfg.pretrained}")

    def embed(self, image: Any) -> list[float]:
        """L2-znormalizowany embedding obrazu (pusta lista, gdy model zwrócił wektor zerowy)."""
        img = _to_pil(image)
        with self._lock:
            vec = np.asarray(self._encode(img), dtype=np.float64).ravel()
        norm = float(np.linalg.norm(vec)) if vec.size else 0.0
        if norm == 0.0 or not np.isfinite(norm):
            return []
        return [round(float(x), 6) for x in vec / norm]

    def _encode(self, img: Any) -> np.ndarray:
        import torch

        with torch.no_grad():
            x = self._preprocess(img).unsqueeze(0).to(self.device)
            feat = self._model.encode_image(x)
        return feat[0].float().cpu().numpy()
