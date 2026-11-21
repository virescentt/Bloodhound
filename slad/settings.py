"""Konfiguracja: config/settings.yaml + zmienne środowiskowe (.env).

Użycie:  from slad.settings import load_settings;  cfg = load_settings()
Ścieżki względne w konfiguracji są liczone względem katalogu głównego repozytorium (ROOT).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


class NetworkSettings(BaseModel):
    enabled: bool = True  # False = tryb offline: tylko dane z wejścia, meta["infra"] i cache
    timeout_s: float = 8.0
    max_bytes: int = 3_000_000
    max_redirects: int = 10
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0 Safari/537.36"
    )
    dns: bool = True
    rdap: bool = True
    tls: bool = True
    favicon: bool = True
    crtsh: bool = False  # Certificate Transparency (crt.sh) — wolne, domyślnie wyłączone
    cache_ttl_h: float = 72.0


class OCRSettings(BaseModel):
    enabled: bool = True
    max_side: int = 2000  # większe obrazy są zmniejszane przed OCR


class LLMSettings(BaseModel):
    enabled: bool = True  # bez klucza w .env → heurystyki (gazetteer + reguły narracji)
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float = 60.0
    max_chars: int = 4000  # ile tekstu materiału wysyłamy do modelu


class FaceSettings(BaseModel):
    enabled: bool = True
    detector_model: str = "models/face_detection_yunet_2023mar.onnx"
    recognizer_model: str = "models/face_recognition_sface_2021dec.onnx"
    gallery_dir: str = "reference/faces"  # reference/faces/<Imię_Nazwisko>/*.jpg
    min_face_score: float = 0.8
    match_threshold: float = 0.363  # próg kosinusowy SFace zalecany przez OpenCV


class ClipSettings(BaseModel):
    enabled: bool = False  # wymaga requirements-extra.txt (open_clip + torch)
    model: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"


class TextSettings(BaseModel):
    minhash_perm: int = 128
    shingle: int = 3  # długość shingla w słowach
    keywords: int = 12


class ClusterSettings(BaseModel):
    method: str = "graph"  # graph = spójne składowe silnych powiązań + Louvain | hdbscan
    resolution: float = 1.0
    min_size: int = 2


class AssignSettings(BaseModel):
    min_score: float = 4.0
    top_k: int = 10


class LinkSettings(BaseModel):
    weights: dict[str, float] = Field(default_factory=dict)
    rarity: bool = True  # ważenie rzadkością (IDF) wartości wskaźników
    thresholds: dict[str, float] = Field(default_factory=lambda: {"duplicate": 12.0, "variant": 7.0, "related": 4.0})
    cluster: ClusterSettings = Field(default_factory=ClusterSettings)
    assign: AssignSettings = Field(default_factory=AssignSettings)


class Settings(BaseModel):
    data_root: str = "data"
    workspace: str = "workspace/default"
    workers: int = 4
    network: NetworkSettings = Field(default_factory=NetworkSettings)
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    faces: FaceSettings = Field(default_factory=FaceSettings)
    clip: ClipSettings = Field(default_factory=ClipSettings)
    text: TextSettings = Field(default_factory=TextSettings)
    link: LinkSettings = Field(default_factory=LinkSettings)

    def path(self, p: str | Path) -> Path:
        """Ścieżka z konfiguracji → absolutna (względem ROOT)."""
        p = Path(p)
        return p if p.is_absolute() else ROOT / p


def _env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    llm = cfg.setdefault("llm", {})
    for key, env in (("base_url", "LLM_BASE_URL"), ("api_key", "LLM_API_KEY"), ("model", "LLM_MODEL")):
        if os.getenv(env):
            llm[key] = os.environ[env]
    if os.getenv("LLM_EXTRA_BODY"):
        try:
            llm["extra_body"] = json.loads(os.environ["LLM_EXTRA_BODY"])
        except json.JSONDecodeError:
            pass
    if os.getenv("SLAD_OFFLINE", "").lower() in {"1", "true", "yes"}:
        cfg.setdefault("network", {})["enabled"] = False
    if os.getenv("SLAD_WORKSPACE"):
        cfg["workspace"] = os.environ["SLAD_WORKSPACE"]
    return cfg


def load_settings(path: str | Path | None = None, **overrides: Any) -> Settings:
    """Wczytuje config/settings.yaml (lub podany plik), nakłada .env i jawne nadpisania."""
    load_dotenv(ROOT / ".env")
    file = Path(path) if path else CONFIG_DIR / "settings.yaml"
    data: dict[str, Any] = yaml.safe_load(file.read_text(encoding="utf-8")) if file.exists() else {}
    data = _env_overrides(data or {})
    data.update(overrides)
    return Settings.model_validate(data)
