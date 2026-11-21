"""Cechy obrazu: hashe percepcyjne (pHash/dHash/wHash/colorHash), kolory dominujące, twarze, CLIP.

Wczytanie jest odporne na różnice formatu: orientacja EXIF, pierwsza klatka animacji (GIF/WebP/APNG),
przezroczystość (kompozycja na białym tle), obrazy 16-bitowe/CMYK, ucięte pliki (wczytanie częściowe).
Nieczytelny plik → ValueError("nieczytelny obraz: …") — orkiestrator zapisuje to w FeatureSet.errors.

Hashe liczymy na wspólnej kopii roboczej (≤ 256 px, bez jednolitych ramek), więc ten sam kreatyw
w innej rozdzielczości, kompresji albo z innym marginesem dostaje (prawie) ten sam hash. Obraz jednolity
(pusty zrzut, biała strona) dostaje PUSTE hashe — inaczej wszystkie puste obrazy byłyby „duplikatami”.

Progi (64-bitowy pHash, odległość Hamminga): ≤ PHASH_DUPLICATE — ta sama grafika,
≤ PHASH_VARIANT — przerobiona grafika; niezwiązane obrazy mają zwykle ~32 ± 4.
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
from PIL import Image, ImageChops, ImageFile, ImageOps

from slad.models import Face, ImageFeatures

log = logging.getLogger(__name__)

MAX_FILE_BYTES = 64 * 1024 * 1024
WORK_SIDE = 256  # kopia robocza do hashy i kolorów
COLOR_SIDE = 64  # miniatura do kolorów dominujących
N_COLORS = 5
WHASH_SCALE = 64  # stała skala wHash → niezależność od rozdzielczości
BLANK_STD = 3.0  # odchylenie jasności (0–255) poniżej którego obraz uznajemy za jednolity
TRIM_TOLERANCE = 16  # różnica od koloru narożnika, poniżej której piksel należy do ramki
TRIM_MIN_AREA = 0.2  # przycinamy ramkę tylko, gdy zostaje ≥ 20% obrazu

PHASH_DUPLICATE = 6
PHASH_VARIANT = 12

_HASH_KINDS = ("phash", "dhash", "whash")
_truncated_lock = threading.Lock()


# ------------------------------------------------------------------ wczytanie


def _decode(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.seek(0)  # pierwsza klatka animacji / pierwsza strona
    img.load()
    return img


def open_image_bytes(data: bytes, name: str = "obraz") -> Image.Image:
    """Bajty → obraz RGB (orientacja EXIF zastosowana, pierwsza klatka, przezroczystość na białym)."""
    if not data:
        raise ValueError(f"nieczytelny obraz: pusty plik {name}")
    try:
        try:
            img = _decode(data)
        except OSError as e:
            if "truncated" not in str(e).lower():
                raise
            # ucięty plik (np. przerwane pobieranie) — czytamy tyle, ile się da
            with _truncated_lock:
                previous = ImageFile.LOAD_TRUNCATED_IMAGES
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                try:
                    img = _decode(data)
                finally:
                    ImageFile.LOAD_TRUNCATED_IMAGES = previous
            log.info("Obraz %s jest ucięty — wczytano częściowo", name)
        try:
            img = ImageOps.exif_transpose(img) or img
        except Exception as e:  # noqa: BLE001 — uszkodzony EXIF nie przekreśla obrazu
            log.debug("EXIF %s: %s", name, e)
        rgb = to_rgb(img)
    except Exception as e:  # noqa: BLE001 — każdy błąd dekodera → ValueError z kontraktu
        raise ValueError(f"nieczytelny obraz: {name}: {e}") from e
    if rgb.width < 1 or rgb.height < 1:
        raise ValueError(f"nieczytelny obraz: {name}: zerowy rozmiar")
    return rgb


def load_image(path: Path) -> Image.Image:
    """Plik → obraz RGB (jak open_image_bytes); brak / nieczytelny plik → ValueError."""
    return open_image_bytes(_read_bytes(Path(path)), Path(path).name)


def _read_bytes(path: Path) -> bytes:
    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ValueError(f"nieczytelny obraz: {path.name}: plik za duży ({size} B)")
        return path.read_bytes()
    except OSError as e:
        raise ValueError(f"nieczytelny obraz: {path.name}: {e.strerror or e}") from e


def to_rgb(img: Image.Image) -> Image.Image:
    """Dowolny tryb PIL → RGB; przezroczystość komponowana na białym tle."""
    if img.mode in ("I", "F") or img.mode.startswith("I;16"):
        arr = np.nan_to_num(np.asarray(img, dtype=np.float64))
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(arr)
        img = Image.fromarray(arr.astype(np.uint8), "L")
    if img.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        canvas.alpha_composite(rgba)
        return canvas.convert("RGB")
    return img if img.mode == "RGB" else img.convert("RGB")


# ------------------------------------------------------------------ cechy


def working_copy(img: Image.Image) -> Image.Image:
    """Kopia ≤ WORK_SIDE px bez jednolitych ramek (letterbox, marginesy zrzutu).

    Wszystkie hashe i tak zmniejszają obraz do ≤ 64 px, więc mała kopia nie traci informacji, a skraca czas.
    """
    work = img.copy()
    if max(work.size) > WORK_SIDE:
        work.thumbnail((WORK_SIDE, WORK_SIDE), Image.Resampling.BOX)
    return _trim_borders(work)


def _trim_borders(img: Image.Image) -> Image.Image:
    """Przycina jednolitą ramkę w kolorze lewego górnego narożnika (różnica = maksimum po kanałach)."""
    if min(img.size) < 8:
        return img
    r, g, b = ImageChops.difference(img, Image.new("RGB", img.size, img.getpixel((0, 0)))).split()
    channel_max = ImageChops.lighter(ImageChops.lighter(r, g), b)
    bbox = channel_max.point(lambda v: 255 if v > TRIM_TOLERANCE else 0).getbbox()
    if not bbox:
        return img
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    area = w * h / (img.width * img.height)
    if area >= 0.98 or area < TRIM_MIN_AREA or min(w, h) < 8:
        return img
    return img.crop(bbox)


def is_blank(img: Image.Image) -> bool:
    """Obraz (prawie) jednolity — hashe nie niosą informacji."""
    return float(np.asarray(img.convert("L"), dtype=np.float32).std()) < BLANK_STD


def image_hashes(work: Image.Image) -> dict[str, str]:
    """pHash/dHash/wHash (64 bity, 16 znaków hex) i colorHash (42 bity, 11 znaków hex)."""
    if is_blank(work):
        return {"phash": "", "dhash": "", "whash": "", "colorhash": ""}
    return {
        "phash": str(imagehash.phash(work)),
        "dhash": str(imagehash.dhash(work)),
        "whash": str(imagehash.whash(work, image_scale=WHASH_SCALE)),
        "colorhash": str(imagehash.colorhash(work)),
    }


def dominant_colors(img: Image.Image, n: int = N_COLORS) -> list[str]:
    """Do n kolorów "#rrggbb" od najczęstszego (median cut na miniaturze — deterministyczne)."""
    small = img.resize((COLOR_SIDE, COLOR_SIDE), Image.Resampling.BILINEAR) if max(img.size) > COLOR_SIDE else img
    quant = small.quantize(colors=max(n, 8), method=Image.Quantize.MEDIANCUT)
    palette = quant.getpalette() or []
    out: list[str] = []
    for _count, idx in sorted(quant.getcolors() or [], key=lambda c: (-c[0], c[1])):
        r, g, b = palette[idx * 3 : idx * 3 + 3]
        color = f"#{r:02x}{g:02x}{b:02x}"
        if color not in out:
            out.append(color)
        if len(out) == n:
            break
    return out


def image_features(path: Path, ctx: Any = None) -> ImageFeatures:
    """Cechy jednego obrazu. Nieczytelny plik → ValueError; błąd twarzy/CLIP → puste pole + ostrzeżenie."""
    path = Path(path)
    data = _read_bytes(path)
    img = open_image_bytes(data, path.name)
    work = working_copy(img)
    return ImageFeatures(
        path=str(path.absolute()),
        sha256=hashlib.sha256(data).hexdigest(),
        width=img.width,
        height=img.height,
        **image_hashes(work),
        dominant_colors=dominant_colors(work),
        faces=_faces(ctx, img, path.name),
        clip=_clip(ctx, img, path.name),
    )


def _faces(ctx: Any, img: Image.Image, name: str) -> list[Face]:
    engine = getattr(ctx, "faces", None)
    if engine is None:
        return []
    try:
        return list(engine.detect(img))
    except Exception as e:  # noqa: BLE001 — hashe obrazu są cenniejsze niż twarze
        log.warning("Detekcja twarzy nie powiodła się (%s): %s", name, e)
        return []


def _clip(ctx: Any, img: Image.Image, name: str) -> list[float]:
    embedder = getattr(ctx, "clip", None)
    if embedder is None:
        return []
    try:
        return [float(v) for v in embedder.embed(img)]
    except Exception as e:  # noqa: BLE001
        log.warning("CLIP nie powiódł się (%s): %s", name, e)
        return []


# ------------------------------------------------------------------ porównania


def hamming(hex_a: str, hex_b: str) -> int:
    """Odległość Hamminga dwóch hashy hex tej samej długości (inaczej ValueError)."""
    if not hex_a or not hex_b or len(hex_a) != len(hex_b):
        raise ValueError(f"nieporównywalne hashe: {hex_a!r} / {hex_b!r}")
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()


def hash_similarity(a: str | None, b: str | None) -> float:
    """1 − hamming / liczba bitów, w [0, 1]; brakujący lub błędny hash → 0.0 (nigdy nie rzuca)."""
    try:
        return 1.0 - hamming(a or "", b or "") / (4 * len(a or ""))
    except (ValueError, TypeError):
        return 0.0


def phash_similarity(a: str | None, b: str | None) -> float:
    """Podobieństwo pHash w [0, 1] (1 = identyczne). 64 bity: 0.91 ≈ PHASH_DUPLICATE, 0.81 ≈ PHASH_VARIANT."""
    return hash_similarity(a, b)


def visual_similarity(a: ImageFeatures, b: ImageFeatures) -> float:
    """Średnie podobieństwo pHash/dHash/wHash dwóch obrazów (0.0, gdy któryś nie ma hashy).

    Trzy niezależne hashe muszą się zgadzać jednocześnie — mniej przypadkowych zbieżności niż sam pHash.
    """
    if a.sha256 and a.sha256 == b.sha256:
        return 1.0 if a.phash else 0.0
    sims = [hash_similarity(getattr(a, k), getattr(b, k)) for k in _HASH_KINDS]
    return sum(sims) / len(sims)
