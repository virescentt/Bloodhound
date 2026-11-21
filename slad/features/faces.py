"""Twarze: detekcja (OpenCV YuNet), embedding (OpenCV SFace) i identyfikacja z bazą referencyjną.

Baza referencyjna: reference/faces/<Imię_Nazwisko>/*.jpg|png|webp — dla każdego zdjęcia bierzemy
największą twarz, osobę reprezentuje średnia L2-znormalizowanych embeddingów. Embeddingi zdjęć są
zapamiętywane w <gallery_dir>/.gallery_cache.json (klucz = SHA256 pliku), więc kolejne uruchomienia
nie liczą ich ponownie. Tożsamość = nazwa katalogu ("_" → spacja), gdy kosinus ≥ match_threshold
(0.363 — próg SFace zalecany przez OpenCV).

Obiekty OpenCV dnn nie są bezpieczne wątkowo — wywołania sieci chroni blokada (workers > 1).
Modele: python -m slad models (slad/features/model_download.py).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from slad.models import Face
from slad.settings import ROOT

if TYPE_CHECKING:
    from slad.settings import FaceSettings

log = logging.getLogger(__name__)

GALLERY_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
CACHE_NAME = ".gallery_cache.json"
CACHE_VERSION = 1
MIN_FACE_SIZE = 40  # px (w oryginale) — mniejsze twarze dają niewiarygodne embeddingi
MAX_SIDE = 1600  # większe obrazy zmniejszamy do detekcji (współrzędne wracają do oryginału)
MAX_FACES = 20  # największe twarze na obraz (tłum na zdjęciu nie blokuje potoku)
# Norma surowego wektora SFace mierzy „jakość” twarzy: zdjęcia (także rozmyte, małe, JPEG q10, szarości)
# ≥ 7, rysunki/awatary ~4–9 (mediana 5.4) — a ich embeddingi są do siebie podobne niezależnie od „osoby”.
# Poniżej progu twarz zostaje (bbox, wynik), ale bez embeddingu i tożsamości → nie tworzy fałszywych przesłanek.
MIN_FEATURE_NORM = 6.0
NMS_THRESHOLD = 0.3
TOP_K = 5000
MODELS_HINT = "uruchom: python -m slad models"


# ------------------------------------------------------------------ pomocnicze


def cosine(a: Sequence[float] | np.ndarray | None, b: Sequence[float] | np.ndarray | None) -> float:
    """Kosinus dwóch wektorów w [-1, 1]; pusty, zerowy lub niezgodny wymiar → 0.0."""
    if a is None or b is None:
        return 0.0
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    if va.size == 0 or va.size != vb.size:
        return 0.0
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0 or not np.isfinite(na * nb):
        return 0.0
    return float(np.clip(va @ vb / (na * nb), -1.0, 1.0))


def l2_normalize(v: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(arr))
    return arr / n if n > 0 and np.isfinite(n) else arr


def person_name(folder: str) -> str:
    """Nazwa katalogu bazy → nazwa osoby: "Jan_Kowalski" → "Jan Kowalski" (Unicode NFC)."""
    name = re.sub(r"\s+", " ", folder.replace("_", " ")).strip()
    return unicodedata.normalize("NFC", name)


def to_bgr(image: Any) -> np.ndarray:
    """PIL.Image (dowolny tryb) albo np.ndarray (konwencja OpenCV: BGR / BGRA / skala szarości) → BGR uint8."""
    import cv2
    from PIL import Image

    if isinstance(image, Image.Image):
        from slad.features.visual import to_rgb

        return cv2.cvtColor(np.asarray(to_rgb(image)), cv2.COLOR_RGB2BGR)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(np.nan_to_num(arr), 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return np.ascontiguousarray(arr)
    raise ValueError(f"nieobsługiwany kształt obrazu: {arr.shape}")


def _resolve(p: str | Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else ROOT / p


def _clip_bbox(row: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0 = max(0, int(round(row[0]))), max(0, int(round(row[1])))
    x1, y1 = min(w, int(round(row[0] + row[2]))), min(h, int(round(row[1] + row[3])))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _as_list(v: np.ndarray | None) -> list[float]:
    return [] if v is None else [round(float(x), 6) for x in v]


# ------------------------------------------------------------------ silnik


class FaceEngine:
    """Detekcja + embedding + identyfikacja twarzy. `detector`/`recognizer` — obiekty cv2 (lub atrapy w testach)."""

    def __init__(
        self,
        detector: Any,
        recognizer: Any,
        *,
        min_face_score: float = 0.8,
        match_threshold: float = 0.363,
        min_face_size: int = MIN_FACE_SIZE,
        max_side: int = MAX_SIDE,
        max_faces: int = MAX_FACES,
        min_feature_norm: float = MIN_FEATURE_NORM,
        model_key: str = "custom",
    ) -> None:
        self._detector = detector
        self._recognizer = recognizer
        self.min_face_score = float(min_face_score)
        self.match_threshold = float(match_threshold)
        self.min_face_size = int(min_face_size)
        self.max_side = int(max_side)
        self.max_faces = int(max_faces)
        self.min_feature_norm = float(min_feature_norm)
        self.model_key = model_key
        self.gallery_dir: Path | None = None
        self._lock = threading.Lock()
        self._gallery: tuple[list[str], np.ndarray] = ([], np.zeros((0, 0), dtype=np.float32))
        self._gallery_counts: dict[str, int] = {}

    # -------------------------------------------------------------- tworzenie

    @classmethod
    def from_models(
        cls,
        detector_model: str | Path,
        recognizer_model: str | Path,
        *,
        min_face_score: float = 0.8,
        nms_threshold: float = NMS_THRESHOLD,
        top_k: int = TOP_K,
        **kwargs: Any,
    ) -> FaceEngine:
        import cv2

        from slad.features.model_download import quiet_opencv

        det_path, rec_path = Path(detector_model), Path(recognizer_model)
        with quiet_opencv():
            detector = cv2.FaceDetectorYN.create(
                str(det_path), "", (320, 320), float(min_face_score), float(nms_threshold), int(top_k)
            )
            recognizer = cv2.FaceRecognizerSF.create(str(rec_path), "")
        key = f"{det_path.name}:{det_path.stat().st_size}|{rec_path.name}:{rec_path.stat().st_size}"
        return cls(detector, recognizer, min_face_score=min_face_score, model_key=key, **kwargs)

    @classmethod
    def from_settings(cls, cfg: FaceSettings) -> FaceEngine | None:
        """Silnik według konfiguracji; None (z podpowiedzią w logu), gdy wyłączony albo brak modeli."""
        if not cfg.enabled:
            log.info("Rozpoznawanie twarzy wyłączone (faces.enabled: false)")
            return None
        det, rec = _resolve(cfg.detector_model), _resolve(cfg.recognizer_model)
        missing = [p.name for p in (det, rec) if not p.is_file()]
        if missing:
            log.warning("Brak modeli twarzy (%s) — %s", ", ".join(missing), MODELS_HINT)
            return None
        try:
            engine = cls.from_models(
                det,
                rec,
                min_face_score=cfg.min_face_score,
                nms_threshold=float(getattr(cfg, "nms_threshold", NMS_THRESHOLD)),
                top_k=int(getattr(cfg, "top_k", TOP_K)),
                match_threshold=cfg.match_threshold,
                min_face_size=int(getattr(cfg, "min_face_size", MIN_FACE_SIZE)),
                max_side=int(getattr(cfg, "max_side", MAX_SIDE)),
                min_feature_norm=float(getattr(cfg, "min_feature_norm", MIN_FEATURE_NORM)),
            )
        except Exception as e:  # noqa: BLE001 — uszkodzony plik modelu
            log.warning("Nie udało się wczytać modeli twarzy (%s) — %s", e, MODELS_HINT)
            return None
        try:
            engine.load_gallery(_resolve(cfg.gallery_dir))
        except Exception as e:  # noqa: BLE001 — bez bazy działa detekcja i porównywanie twarzy
            log.warning("Baza twarzy niedostępna (%s): %s", cfg.gallery_dir, e)
        return engine

    # -------------------------------------------------------------- detekcja

    def detect(self, image: Any) -> list[Face]:
        """Twarze na obrazie (od największej): bbox w pikselach oryginału, embedding L2, tożsamość z bazy."""
        bgr = to_bgr(image)
        h, w = bgr.shape[:2]
        if min(h, w) < self.min_face_size:
            return []
        with self._lock:
            found = [(row, self._embed(bgr, row)) for row in self._detect_rows(bgr)]
        faces: list[Face] = []
        for row, emb in found:
            identity, identity_score = self.identify(emb) if emb is not None else (None, None)
            faces.append(
                Face(
                    bbox=_clip_bbox(row, w, h),
                    score=round(float(row[14]), 4),
                    embedding=_as_list(emb),
                    identity=identity,
                    identity_score=identity_score,
                )
            )
        return faces

    def _detect_rows(self, bgr: np.ndarray) -> list[np.ndarray]:
        """Wiersze YuNet (15 liczb, współrzędne oryginału) po filtrach, od największej twarzy. Wymaga blokady."""
        import cv2

        h, w = bgr.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        img = bgr
        if scale < 1.0:
            img = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        self._detector.setInputSize((img.shape[1], img.shape[0]))
        _, raw = self._detector.detect(img)
        if raw is None or len(raw) == 0:
            return []
        rows: list[np.ndarray] = []
        for r in np.asarray(raw, dtype=np.float32).reshape(-1, 15):
            if not np.all(np.isfinite(r)) or float(r[14]) < self.min_face_score:
                continue
            row = r.copy()
            row[:14] /= scale
            if min(row[2], row[3]) < self.min_face_size:
                continue
            rows.append(row)
        rows.sort(key=lambda r: float(r[2] * r[3]), reverse=True)
        return rows[: self.max_faces]

    def _embed(self, bgr: np.ndarray, row: np.ndarray) -> np.ndarray | None:
        """alignCrop + SFace → wektor L2. None (twarz zostaje bez embeddingu), gdy błąd albo niska jakość
        (norma surowego wektora < min_feature_norm — rysunek, awatar, twarz nieczytelna). Wymaga blokady."""
        try:
            aligned = self._recognizer.alignCrop(bgr, row.reshape(1, 15))
            raw = np.asarray(self._recognizer.feature(aligned), dtype=np.float32).ravel()
        except Exception as e:  # noqa: BLE001
            log.debug("Embedding twarzy nie powiódł się: %s", e)
            return None
        norm = float(np.linalg.norm(raw)) if raw.size else 0.0
        if not np.isfinite(norm) or norm == 0.0:
            return None
        if norm < self.min_feature_norm:
            log.debug("Twarz niskiej jakości (norma %.1f < %.1f) — bez embeddingu", norm, self.min_feature_norm)
            return None
        return raw / norm

    # -------------------------------------------------------------- identyfikacja

    @property
    def people(self) -> list[str]:
        """Osoby w bazie referencyjnej (z co najmniej jednym zdjęciem z twarzą)."""
        return list(self._gallery[0])

    @property
    def gallery_counts(self) -> dict[str, int]:
        """Osoba → liczba zdjęć referencyjnych z wykrytą twarzą."""
        return dict(self._gallery_counts)

    def set_gallery(self, embeddings: dict[str, Sequence[Sequence[float]] | Sequence[float]]) -> None:
        """Ustawia bazę z gotowych embeddingów: osoba → wektor albo lista wektorów (uśredniana)."""
        names: list[str] = []
        rows: list[np.ndarray] = []
        counts: dict[str, int] = {}
        for name, vecs in sorted(embeddings.items()):
            arr = np.asarray(vecs, dtype=np.float32)
            arr = arr.reshape(1, -1) if arr.ndim == 1 else arr.reshape(arr.shape[0], -1)
            if arr.size == 0:
                continue
            mean = l2_normalize(np.mean([l2_normalize(v) for v in arr], axis=0))
            if float(np.linalg.norm(mean)) == 0.0:
                continue
            names.append(name)
            rows.append(mean)
            counts[name] = int(arr.shape[0])
        if rows and len({r.size for r in rows}) != 1:
            raise ValueError("embeddingi bazy mają różne wymiary")
        matrix = np.stack(rows) if rows else np.zeros((0, 0), dtype=np.float32)
        self._gallery = (names, matrix)  # podmiana atomowa — bezpieczna dla wątków czytających
        self._gallery_counts = counts

    def identify(self, embedding: Sequence[float] | np.ndarray | None) -> tuple[str | None, float | None]:
        """(osoba, kosinus) najbliższej osoby z bazy, gdy kosinus ≥ match_threshold; inaczej (None, None)."""
        names, matrix = self._gallery
        if not names or embedding is None:
            return None, None
        v = l2_normalize(embedding)
        if v.size != matrix.shape[1] or float(np.linalg.norm(v)) == 0.0:
            return None, None
        sims = matrix @ v
        i = int(np.argmax(sims))
        score = float(sims[i])
        if score >= self.match_threshold:
            return names[i], round(score, 4)
        return None, None

    # -------------------------------------------------------------- baza referencyjna

    def load_gallery(self, gallery_dir: str | Path) -> dict[str, int]:
        """Wczytuje bazę reference/faces/<Imię_Nazwisko>/*; zwraca osoba → liczba zdjęć z twarzą."""
        gallery_dir = Path(gallery_dir)
        self.gallery_dir = gallery_dir
        if not gallery_dir.is_dir():
            log.info("Brak katalogu bazy twarzy %s — identyfikacja osób wyłączona", gallery_dir)
            self.set_gallery({})
            return {}
        cache_path = gallery_dir / CACHE_NAME
        cache_key = (
            f"{self.model_key}|score>={self.min_face_score}|size>={self.min_face_size}|norm>={self.min_feature_norm}"
        )
        cached = self._read_cache(cache_path, cache_key)
        entries: dict[str, dict[str, Any]] = {}
        people: dict[str, list[list[float]]] = {}
        computed = 0
        for person_dir in sorted(p for p in gallery_dir.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
            name = person_name(person_dir.name)
            if not name:
                continue
            for f in sorted(person_dir.iterdir()):
                if f.name.startswith(".") or f.suffix.lower() not in GALLERY_EXTS or not f.is_file():
                    continue
                rel = f.relative_to(gallery_dir).as_posix()
                try:
                    data = f.read_bytes()
                except OSError as e:
                    log.warning("Baza twarzy: nie można odczytać %s: %s", rel, e)
                    continue
                sha = hashlib.sha256(data).hexdigest()
                entry = cached.get(sha)
                if entry is None:
                    emb = self._reference_embedding(data, rel)
                    entry = {"file": rel, "embedding": _as_list(emb) or None}
                    computed += 1
                entries[sha] = entry
                if entry.get("embedding"):
                    people.setdefault(name, []).append(entry["embedding"])
                else:
                    log.warning("Baza twarzy: nie wykryto twarzy na zdjęciu %s", rel)
        if computed or set(entries) != set(cached):
            self._write_cache(cache_path, cache_key, entries)
        self.set_gallery(people)
        counts = self.gallery_counts
        log.info(
            "Baza twarzy: %d osób, %d zdjęć z twarzą (%d policzonych od nowa)",
            len(counts),
            sum(counts.values()),
            computed,
        )
        return counts

    def _reference_embedding(self, data: bytes, rel: str) -> np.ndarray | None:
        """Embedding największej twarzy na zdjęciu referencyjnym (None, gdy brak twarzy / nieczytelne)."""
        from slad.features.visual import open_image_bytes

        try:
            bgr = to_bgr(open_image_bytes(data, rel))
        except ValueError as e:
            log.warning("Baza twarzy: %s", e)
            return None
        if min(bgr.shape[:2]) < self.min_face_size:
            return None
        with self._lock:
            rows = self._detect_rows(bgr)
            return self._embed(bgr, rows[0]) if rows else None

    @staticmethod
    def _read_cache(path: Path, key: str) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            log.warning("Pamięć bazy twarzy %s nieczytelna — liczę od nowa: %s", path.name, e)
            return {}
        if not isinstance(data, dict) or data.get("version") != CACHE_VERSION or data.get("model") != key:
            return {}
        files = data.get("files")
        return {k: v for k, v in files.items() if isinstance(v, dict)} if isinstance(files, dict) else {}

    @staticmethod
    def _write_cache(path: Path, key: str, entries: dict[str, dict[str, Any]]) -> None:
        payload = {"version": CACHE_VERSION, "model": key, "files": entries}
        try:
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                os.chmod(tmp, 0o644)  # mkstemp tworzy 0600
                os.replace(tmp, path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        except OSError as e:  # katalog tylko do odczytu (np. wolumen w Dockerze) — działamy bez pamięci
            log.warning("Nie można zapisać pamięci bazy twarzy %s: %s", path, e)
