"""OCR zrzutów ekranu i kreacji reklamowych — RapidOCR (PP-OCRv6, ONNX Runtime, CPU; modele w pakiecie pip).

    ocr_image(path, ctx) -> str    linie tekstu w kolejności czytania (z góry na dół, w linii od lewej do prawej)

- Obraz: transpozycja wg EXIF, RGB (przezroczystość na białym tle), zmniejszenie tak, by dłuższy bok
  <= ctx.settings.ocr.max_side. Bardzo wysokie zrzuty (cała strona) są dzielone na fragmenty w pustych wierszach
  zamiast zmniejszania — inaczej tekst stałby się nieczytelny (każdy fragment nadal <= max_side).
- Wynik w ctx.cache("ocr") pod kluczem sha256 pliku — powtórne przetworzenie nie liczy OCR ponownie.
- Silnik ładowany raz na proces (leniwie); inferencja pod blokadą (wątki pipeline'u; ONNX Runtime i tak używa
  wszystkich rdzeni dla jednego obrazu).
- Brak lub awaria RapidOCR → "" i jedno ostrzeżenie w logu (pipeline działa dalej bez OCR).
- Nieczytelny plik → ValueError (orkiestrator zapisze błąd w FeatureSet.errors).
Odczyt wyników RapidOCR 3.x i grupowanie w linie — na wzór odwolania2json/ingest/ocr.py (Odwołania2JSON).
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

OCR_CACHE_VERSION = 1
TALL_RATIO = 2.5  # wysokość / szerokość, od której zrzut traktujemy jak całą stronę (cięcie zamiast zmniejszania)
MAX_TILES = 12

_LOCK = threading.Lock()
_engine: Any = None
_engine_failed = False
_warned = False

Segment = tuple[float, float, float, float, str]  # x0, y0, x1, y1, tekst


def _warn_once(msg: str, *args: Any) -> None:
    global _warned
    if _warned:
        log.debug(msg, *args)
    else:
        _warned = True
        log.warning(msg, *args)


def _get_engine() -> Any:
    """RapidOCR (singleton) albo None, gdy pakiet / modele są niedostępne. Wywoływać pod _LOCK."""
    global _engine, _engine_failed
    if _engine is None and not _engine_failed:
        try:
            from rapidocr import RapidOCR

            _engine = RapidOCR(
                params={
                    "Global.log_level": "error",
                    "Global.use_cls": False,  # tekst obrócony o 180° w reklamach praktycznie nie występuje
                    "Global.max_side_len": 100_000,  # skalujemy sami (max_side), RapidOCR ma nie zmniejszać ponownie
                }
            )
        except Exception as e:  # noqa: BLE001
            _engine_failed = True
            _warn_once("OCR niedostępny (RapidOCR: %s) — obrazy będą przetwarzane bez odczytu tekstu", e)
    return _engine


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image(path: Path) -> Any:
    """Plik → PIL.Image RGB (EXIF, przezroczystość na białym tle). Nieczytelny plik → ValueError."""
    from PIL import Image, ImageOps

    try:
        with Image.open(path) as im:
            im.load()
            img = ImageOps.exif_transpose(im) or im.copy()
    except Exception as e:  # noqa: BLE001 — uszkodzony plik, nie-obraz, „bomba dekompresyjna”…
        raise ValueError(f"nieczytelny obraz {Path(path).name}: {e}") from e
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img if img.mode == "RGB" else img.convert("RGB")


def tiles(img: Any, max_side: int) -> list[Any]:
    """Fragmenty obrazu do OCR, każdy o dłuższym boku <= max_side.

    Zwykły obraz → jeden fragment (zmniejszony, jeśli trzeba). Wysoki zrzut strony → szerokość ograniczona do
    3/4 max_side, potem cięcie na pasy wysokości <= max_side w najbardziej „pustym” wierszu ostatnich 20% pasa
    (zwykle odstęp między liniami tekstu), maks. MAX_TILES pasów.
    """
    from PIL import Image

    max_side = max(64, int(max_side))
    w, h = img.size
    if max(w, h) <= max_side:
        return [img]
    if h <= w * TALL_RATIO:
        s = max_side / max(w, h)
        return [img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.LANCZOS)]

    import numpy as np

    target_w = min(w, int(max_side * 0.75))
    if math.ceil(h * target_w / w / max_side) > MAX_TILES:  # bardzo długa strona — zmniejszamy, żeby się zmieściła
        target_w = max(1, int(w * MAX_TILES * max_side / h))
    if target_w < w:
        img = img.resize((target_w, max(1, round(h * target_w / w))), Image.Resampling.LANCZOS)
        w, h = img.size
    row_ink = np.asarray(img.convert("L"), dtype=np.float32).std(axis=1)  # 0 = wiersz jednolity (odstęp)
    out, top = [], 0
    while top < h:
        bottom = min(h, top + max_side)
        if bottom < h:
            lo = top + int(max_side * 0.8)
            bottom = lo + int(np.argmin(row_ink[lo:bottom]))
        out.append(img.crop((0, top, w, bottom)))
        top = bottom
    return out


def read_segments(engine: Any, img: Any) -> list[Segment]:
    """Wynik RapidOCR 3.x (boxes: czworokąty, txts: teksty) → prostokąty z tekstem."""
    res = engine(img)
    txts = getattr(res, "txts", None) or ()
    boxes = getattr(res, "boxes", None)
    if not txts or boxes is None:
        return []
    out: list[Segment] = []
    for quad, txt in zip(boxes, txts, strict=False):
        text = " ".join(str(txt).split())
        if not text:
            continue
        xs = [float(p[0]) for p in quad]
        ys = [float(p[1]) for p in quad]
        out.append((min(xs), min(ys), max(xs), max(ys), text))
    return out


def group_lines(segments: list[Segment]) -> list[str]:
    """Segmenty → linie tekstu: grupowanie po środku w pionie (tolerancja: pół wysokości), w linii od lewej."""
    lines: list[dict[str, Any]] = []
    for x0, y0, _x1, y1, text in sorted(segments, key=lambda s: (s[1] + s[3]) / 2):
        cy, height = (y0 + y1) / 2, max(1.0, y1 - y0)
        best = None
        for line in lines[-4:]:
            d = abs(cy - line["cy"])
            if d <= 0.5 * min(height, line["h"]) and (best is None or d < best[0]):
                best = (d, line)
        if best is None:
            lines.append({"cy": cy, "h": height, "segs": [(x0, text)]})
            continue
        line = best[1]
        n = len(line["segs"])
        line["cy"] = (line["cy"] * n + cy) / (n + 1)
        line["h"] = (line["h"] * n + height) / (n + 1)
        line["segs"].append((x0, text))
    lines.sort(key=lambda ln: ln["cy"])
    return [" ".join(t for _, t in sorted(ln["segs"], key=lambda s: s[0])) for ln in lines]


def _cache(ctx: Any) -> Any:
    try:
        return ctx.cache("ocr")
    except Exception as e:  # noqa: BLE001 — bez cache też działa
        log.debug("Cache OCR niedostępny: %s", e)
        return None


def _cache_get(cache: Any, key: str) -> Any:
    if cache is None:
        return None
    try:
        return cache.get(key)
    except Exception as e:  # noqa: BLE001
        log.debug("Odczyt cache OCR: %s", e)
        return None


def _cache_set(cache: Any, key: str, value: Any) -> None:
    if cache is None:
        return
    try:
        if hasattr(cache, "set"):
            cache.set(key, value)
        else:
            cache[key] = value
    except Exception as e:  # noqa: BLE001
        log.debug("Zapis cache OCR: %s", e)


def ocr_image(path: Path, ctx: Any) -> str:
    """Tekst z obrazu (linie w kolejności czytania). "" gdy OCR niedostępny lub wyłączony."""
    path = Path(path)
    ocr_cfg = getattr(getattr(ctx, "settings", None), "ocr", None)
    if ocr_cfg is not None and not getattr(ocr_cfg, "enabled", True):
        return ""
    max_side = int(getattr(ocr_cfg, "max_side", 2000) or 2000)
    try:
        digest = file_sha256(path)
    except OSError as e:
        raise ValueError(f"brak dostępu do pliku {path.name}: {e}") from e

    cache = _cache(ctx)
    hit = _cache_get(cache, digest)
    if isinstance(hit, dict) and hit.get("v") == OCR_CACHE_VERSION and hit.get("max_side") == max_side:
        if isinstance(hit.get("text"), str):
            return hit["text"]

    img = load_image(path)
    with _LOCK:
        engine = _get_engine()
        if engine is None:
            return ""
        try:
            lines: list[str] = []
            for tile in tiles(img, max_side):
                lines.extend(group_lines(read_segments(engine, tile)))
        except Exception as e:  # noqa: BLE001
            _warn_once("OCR nie powiódł się (%s: %s) — pomijam odczyt tekstu", path.name, e)
            return ""
    text = "\n".join(lines)
    _cache_set(cache, digest, {"v": OCR_CACHE_VERSION, "engine": "rapidocr", "max_side": max_side, "text": text})
    return text
