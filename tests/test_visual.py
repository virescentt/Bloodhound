"""Testy cech obrazu (slad.features.visual) i opcjonalnego CLIP (slad.features.clip) — bez sieci."""

from __future__ import annotations

import contextlib
import io
import logging
import re
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageEnhance, features

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.features.clip import ClipEmbedder  # noqa: E402
from slad.features.visual import (  # noqa: E402
    PHASH_DUPLICATE,
    PHASH_VARIANT,
    WORK_SIDE,
    dominant_colors,
    hamming,
    hash_similarity,
    image_features,
    load_image,
    phash_similarity,
    visual_similarity,
    working_copy,
)
from slad.models import Face, ImageFeatures  # noqa: E402
from slad.settings import ClipSettings  # noqa: E402

# ------------------------------------------------------------------ pomocnicze


def make_image(seed: int, size: tuple[int, int] = (480, 360)) -> Image.Image:
    """Syntetyczna „kreacja”: gładkie tło o niskiej częstotliwości + kilka figur (deterministyczne)."""
    rng = np.random.default_rng(seed)
    small = (rng.random((6, 8, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(small).resize(size, Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(img)
    for _ in range(6):
        x0, y0 = int(rng.integers(0, size[0] - 60)), int(rng.integers(0, size[1] - 60))
        x1, y1 = x0 + int(rng.integers(30, 200)), y0 + int(rng.integers(30, 150))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        (draw.rectangle if rng.random() < 0.5 else draw.ellipse)([x0, y0, x1, y1], fill=color)
    return img


def save(img: Image.Image, path: Path, fmt: str | None = None, **kw) -> Path:
    img.save(path, fmt, **kw)
    return path


def ctx(faces=None, clip=None) -> types.SimpleNamespace:
    return types.SimpleNamespace(faces=faces, clip=clip)


def feats(img: Image.Image, path: Path, fmt: str | None = None, **kw) -> ImageFeatures:
    return image_features(save(img, path, fmt, **kw), ctx())


@pytest.fixture
def base() -> Image.Image:
    return make_image(1)


# ------------------------------------------------------------------ podobieństwo


def test_identical_images_have_identical_features(tmp_path: Path, base: Image.Image) -> None:
    a = feats(base, tmp_path / "a.png")
    b = feats(base, tmp_path / "b.png")
    assert (a.phash, a.dhash, a.whash, a.colorhash) == (b.phash, b.dhash, b.whash, b.colorhash)
    assert a.sha256 == b.sha256 and len(a.sha256) == 64
    assert len(a.phash) == len(a.dhash) == len(a.whash) == 16
    assert hamming(a.phash, b.phash) == 0
    assert phash_similarity(a.phash, b.phash) == 1.0
    assert visual_similarity(a, b) == 1.0
    assert (a.width, a.height) == base.size
    assert Path(a.path).is_absolute() and a.path.endswith("a.png")


def test_recompression_and_resize_are_duplicates(tmp_path: Path, base: Image.Image) -> None:
    ref = feats(base, tmp_path / "ref.png")
    jpeg = feats(base, tmp_path / "q60.jpg", "JPEG", quality=60)
    half = feats(base.resize((base.width // 2, base.height // 2), Image.Resampling.LANCZOS), tmp_path / "half.png")
    for variant in (jpeg, half):
        assert hamming(ref.phash, variant.phash) <= PHASH_DUPLICATE
        assert visual_similarity(ref, variant) >= 0.85
    assert jpeg.sha256 != ref.sha256


def test_brightness_and_small_crop_stay_close(tmp_path: Path, base: Image.Image) -> None:
    ref = feats(base, tmp_path / "ref.png")
    bright = feats(ImageEnhance.Brightness(base).enhance(1.15), tmp_path / "bright.png")
    w, h = base.size
    crop = feats(base.crop((int(w * 0.03), int(h * 0.03), int(w * 0.97), int(h * 0.97))), tmp_path / "crop.png")
    assert hamming(ref.phash, bright.phash) <= 8
    assert hamming(ref.phash, crop.phash) <= PHASH_VARIANT


def test_letterboxed_copy_matches_original(tmp_path: Path, base: Image.Image) -> None:
    """Ta sama kreacja z białym marginesem (inny kadr zrzutu) — ramka jest przycinana przed hashowaniem."""
    ref = feats(base, tmp_path / "ref.png")
    canvas = Image.new("RGB", (base.width + 120, base.height + 60), "white")
    canvas.paste(base, (60, 30))
    boxed = feats(canvas, tmp_path / "boxed.png")
    assert hamming(ref.phash, boxed.phash) <= PHASH_DUPLICATE
    assert (boxed.width, boxed.height) == canvas.size  # rozmiar oryginału, nie kopii roboczej


def test_border_trim_keeps_light_colored_content() -> None:
    img = Image.new("RGB", (200, 150), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 15, 179, 134], fill=(255, 255, 150))  # jasnożółty panel: różni się od bieli tylko w B
    draw.ellipse([75, 50, 125, 100], fill=(30, 30, 30))
    assert working_copy(img).size == (160, 120)  # przycięta tylko biała ramka, panel zostaje
    full = make_image(2, (240, 180))
    assert working_copy(full).size == full.size  # obraz bez ramki — bez zmian
    big = make_image(2, (1600, 1200))
    assert working_copy(big).size == (WORK_SIDE, WORK_SIDE * 3 // 4)  # kopia robocza ≤ WORK_SIDE


def test_different_images_are_far_apart(tmp_path: Path) -> None:
    fs = [feats(make_image(seed), tmp_path / f"{seed}.png") for seed in range(10, 20)]
    pairs = [(a, b) for i, a in enumerate(fs) for b in fs[i + 1 :]]
    assert min(hamming(a.phash, b.phash) for a, b in pairs) >= 16
    assert max(phash_similarity(a.phash, b.phash) for a, b in pairs) < 0.75
    assert max(visual_similarity(a, b) for a, b in pairs) < 0.75


def test_blank_images_get_empty_hashes(tmp_path: Path) -> None:
    """Puste / jednolite obrazy (martwa strona, biały zrzut) nie mogą być „duplikatami” siebie nawzajem."""
    white = feats(Image.new("RGB", (300, 200), "white"), tmp_path / "white.png")
    gray = feats(Image.new("RGB", (640, 480), (128, 128, 128)), tmp_path / "gray.jpg", "JPEG")
    for f in (white, gray):
        assert (f.phash, f.dhash, f.whash, f.colorhash) == ("", "", "", "")
        assert f.sha256 and f.width > 0
    assert white.dominant_colors == ["#ffffff"]
    assert phash_similarity(white.phash, gray.phash) == 0.0
    assert visual_similarity(white, gray) == 0.0
    assert visual_similarity(white, white) == 0.0  # ten sam pusty plik też nie jest przesłanką


# ------------------------------------------------------------------ odporność na format


@pytest.mark.parametrize(
    "content",
    [b"", b"not an image at all", b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, b"<svg xmlns='http://www.w3.org/2000/svg'/>"],
    ids=["empty", "text", "broken-png", "svg"],
)
def test_unreadable_image_raises_value_error(tmp_path: Path, content: bytes) -> None:
    p = tmp_path / "bad.png"
    p.write_bytes(content)
    with pytest.raises(ValueError, match="^nieczytelny obraz"):
        image_features(p, ctx())


def test_missing_file_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="^nieczytelny obraz"):
        image_features(tmp_path / "brak.jpg", ctx())
    with pytest.raises(ValueError, match="^nieczytelny obraz"):
        load_image(tmp_path)  # katalog zamiast pliku


def test_truncated_jpeg_is_read_partially(tmp_path: Path, base: Image.Image) -> None:
    buf = io.BytesIO()
    base.save(buf, "JPEG", quality=90)
    data = buf.getvalue()
    p = tmp_path / "cut.jpg"
    p.write_bytes(data[: int(len(data) * 0.7)])
    f = image_features(p, ctx())
    assert (f.width, f.height) == base.size
    assert len(f.phash) == 16


def test_rgba_is_composited_on_white(tmp_path: Path, base: Image.Image) -> None:
    rgba = base.convert("RGBA")
    alpha = Image.new("L", base.size, 255)
    ImageDraw.Draw(alpha).rectangle([0, 0, base.width // 3, base.height], fill=0)  # lewa 1/3 przezroczysta
    rgba.putalpha(alpha)
    expected = Image.new("RGB", base.size, "white")
    expected.paste(base, mask=alpha)
    a = feats(rgba, tmp_path / "rgba.png")
    b = feats(expected, tmp_path / "flat.png")
    assert hamming(a.phash, b.phash) == 0
    assert a.dominant_colors == b.dominant_colors


def test_animated_gif_uses_first_frame(tmp_path: Path) -> None:
    first, second = make_image(3), make_image(4)
    p = tmp_path / "anim.gif"
    first.save(p, save_all=True, append_images=[second], duration=200, loop=0)
    anim = image_features(p, ctx())
    f1 = feats(first, tmp_path / "first.png")
    f2 = feats(second, tmp_path / "second.png")
    assert hamming(anim.phash, f1.phash) <= PHASH_DUPLICATE + 2  # paleta GIF lekko zmienia obraz
    assert hamming(anim.phash, f2.phash) >= 16


@pytest.mark.skipif(not features.check("webp"), reason="Pillow bez WebP")
def test_animated_webp_uses_first_frame(tmp_path: Path) -> None:
    first, second = make_image(5), make_image(6)
    p = tmp_path / "anim.webp"
    first.save(p, save_all=True, append_images=[second], duration=200, loop=0, lossless=True)
    anim = image_features(p, ctx())
    assert hamming(anim.phash, feats(first, tmp_path / "f.png").phash) <= PHASH_DUPLICATE
    assert hamming(anim.phash, feats(second, tmp_path / "s.png").phash) >= 16


def test_exif_orientation_is_applied(tmp_path: Path, base: Image.Image) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # obróć o 90° w prawo przy wyświetlaniu
    p = save(base, tmp_path / "rotated.jpg", "JPEG", quality=95, exif=exif.tobytes())
    f = image_features(p, ctx())
    assert (f.width, f.height) == (base.height, base.width)
    upright = feats(base.transpose(Image.Transpose.ROTATE_270), tmp_path / "upright.png")
    assert hamming(f.phash, upright.phash) <= 2


def test_unusual_modes_do_not_crash(tmp_path: Path, base: Image.Image) -> None:
    rng = np.random.default_rng(0)
    gray16 = Image.fromarray((np.asarray(base.convert("L"), dtype=np.uint16) * 257).astype(np.uint16))
    pal = base.convert("P", palette=Image.Palette.ADAPTIVE, colors=32)
    pal.info["transparency"] = 0
    cases = {
        "gray16.png": (gray16, None, {}),
        "cmyk.jpg": (base.convert("CMYK"), "JPEG", {}),
        "bilevel.png": (base.convert("1"), None, {}),
        "la.png": (base.convert("LA"), None, {}),
        "pal.png": (pal, None, {"transparency": 0}),
        "tiny.png": (Image.fromarray(rng.integers(0, 255, (1, 1, 3), dtype=np.uint8)), None, {}),
        "thin.png": (base.resize((400, 3)), None, {}),
        "tiff.tif": (base, "TIFF", {}),
        "bmp.bmp": (base, "BMP", {}),
    }
    for name, (img, fmt, kw) in cases.items():
        f = image_features(save(img, tmp_path / name, fmt, **kw), ctx())
        assert (f.width, f.height) == img.size, name
        assert isinstance(f.phash, str) and f.dominant_colors, name
    gray = image_features(tmp_path / "gray16.png", ctx())
    ref = feats(base.convert("L"), tmp_path / "gray8.png")
    assert hamming(gray.phash, ref.phash) <= PHASH_DUPLICATE  # 16 bitów przeskalowane do 8, nie „wypalone” do bieli


def test_dominant_colors_are_deterministic(tmp_path: Path) -> None:
    img = Image.new("RGB", (200, 100), (230, 20, 20))
    ImageDraw.Draw(img).rectangle([100, 0, 200, 100], fill=(20, 20, 230))
    ImageDraw.Draw(img).rectangle([0, 80, 40, 100], fill=(20, 200, 20))
    colors = dominant_colors(img)
    assert colors == dominant_colors(img.copy())
    assert 1 <= len(colors) <= 5
    assert all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in colors)
    top = [tuple(int(c[i : i + 2], 16) for i in (1, 3, 5)) for c in colors[:2]]
    assert any(r > 200 and b < 60 for r, _, b in top) and any(b > 200 and r < 60 for r, _, b in top)
    many = feats(make_image(7), tmp_path / "m.png").dominant_colors
    assert len(many) == 5 and len(set(many)) == 5


def test_large_image_is_fast(tmp_path: Path) -> None:
    p = save(make_image(8, (3000, 2400)), tmp_path / "big.jpg", "JPEG", quality=85)
    t0 = time.perf_counter()
    f = image_features(p, ctx())
    assert time.perf_counter() - t0 < 1.5
    assert (f.width, f.height) == (3000, 2400)


def test_load_image_returns_rgb(tmp_path: Path, base: Image.Image) -> None:
    img = load_image(save(base.convert("RGBA"), tmp_path / "x.png"))
    assert img.mode == "RGB" and img.size == base.size


# ------------------------------------------------------------------ twarze i CLIP przez kontekst


class _RecordingFaces:
    def __init__(self) -> None:
        self.seen: list[tuple[str, tuple[int, int]]] = []

    def detect(self, image) -> list[Face]:
        self.seen.append((image.mode, image.size))
        return [
            Face(bbox=(1, 2, 50, 60), score=0.99, embedding=[1.0, 0.0], identity="Jan Kowalski", identity_score=0.9)
        ]


class _BrokenEngine:
    def detect(self, image):
        raise RuntimeError("model padł")

    def embed(self, image):
        raise RuntimeError("model padł")


def test_faces_and_clip_from_context(tmp_path: Path, base: Image.Image) -> None:
    rec = _RecordingFaces()
    clip = types.SimpleNamespace(embed=lambda image: [0.6, 0.8])
    p = save(base.convert("RGBA"), tmp_path / "x.png")
    f = image_features(p, ctx(faces=rec, clip=clip))
    assert rec.seen == [("RGB", base.size)]  # pełna rozdzielczość, RGB
    assert f.faces[0].identity == "Jan Kowalski"
    assert f.clip == [0.6, 0.8]
    assert image_features(p, None).faces == []  # brak kontekstu = brak twarzy i CLIP


def test_failing_face_engine_keeps_hashes(tmp_path: Path, base: Image.Image, caplog) -> None:
    p = save(base, tmp_path / "x.png")
    with caplog.at_level(logging.WARNING):
        f = image_features(p, ctx(faces=_BrokenEngine(), clip=_BrokenEngine()))
    assert f.faces == [] and f.clip == []
    assert len(f.phash) == 16
    assert "twarzy" in caplog.text


# ------------------------------------------------------------------ porównania hashy


def test_hamming_and_similarity() -> None:
    zero, ones = "0" * 16, "f" * 16
    assert hamming(zero, "0" * 15 + "f") == 4
    assert hamming(zero, ones) == 64
    assert phash_similarity(zero, zero) == 1.0
    assert phash_similarity(zero, ones) == 0.0
    assert phash_similarity(zero, "0" * 15 + "f") == pytest.approx(1 - 4 / 64)
    assert hash_similarity("00000000000", "00000000001") == pytest.approx(1 - 1 / 44)  # colorhash (11 hex)
    for bad in ("", None, "0" * 15, "zz" * 8):
        assert phash_similarity(zero, bad) == 0.0
    for a, b in (("", ""), (zero, "0" * 15), (zero, "g" * 16)):
        with pytest.raises(ValueError):
            hamming(a, b)


# ------------------------------------------------------------------ CLIP (bez torch — atrapy)


class _FakeTensor:
    def __init__(self, arr) -> None:
        self.arr = np.asarray(arr, dtype=np.float32)

    def unsqueeze(self, dim: int) -> _FakeTensor:
        return _FakeTensor(np.expand_dims(self.arr, dim))

    def to(self, device: str) -> _FakeTensor:
        return self

    def __getitem__(self, i) -> _FakeTensor:
        return _FakeTensor(self.arr[i])

    def float(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.arr


class _FakeClipModel:
    def eval(self) -> _FakeClipModel:
        return self

    def encode_image(self, x: _FakeTensor) -> _FakeTensor:
        return _FakeTensor(x.arr.reshape(x.arr.shape[0], -1)[:, :6] + 0.5)


def _fake_open_clip(fail: bool = False) -> types.SimpleNamespace:
    def create(name, pretrained=None):
        if fail:
            raise RuntimeError("brak wag offline")
        return _FakeClipModel(), None, lambda img: _FakeTensor(np.asarray(img.resize((2, 2)), dtype=np.float32) / 255)

    return types.SimpleNamespace(create_model_and_transforms=create)


def test_clip_disabled_or_missing_returns_none(monkeypatch) -> None:
    assert ClipEmbedder.from_settings(ClipSettings(enabled=False)) is None
    monkeypatch.setitem(sys.modules, "open_clip", None)  # import → ImportError
    assert ClipEmbedder.from_settings(ClipSettings(enabled=True)) is None


def test_clip_model_load_failure_returns_none(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "open_clip", _fake_open_clip(fail=True))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(no_grad=contextlib.nullcontext))
    assert ClipEmbedder.from_settings(ClipSettings(enabled=True)) is None


def test_clip_embed_with_fake_backend(monkeypatch, tmp_path: Path, base: Image.Image) -> None:
    monkeypatch.setitem(sys.modules, "open_clip", _fake_open_clip())
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(no_grad=contextlib.nullcontext))
    embedder = ClipEmbedder.from_settings(ClipSettings(enabled=True))
    assert embedder is not None
    vec = embedder.embed(base)
    assert len(vec) == 6 and np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)
    assert embedder.embed(np.asarray(base)) == vec  # ndarray RGB (konwencja PIL)
    assert len(embedder.embed(np.zeros((8, 8, 4), dtype=np.uint8))) == 6  # RGBA ndarray
    f = image_features(save(base, tmp_path / "x.png"), ctx(clip=embedder))
    assert f.clip == vec


def test_clip_zero_vector_gives_empty_embedding(base: Image.Image) -> None:
    class Zero(ClipEmbedder):
        def _encode(self, img):
            return np.zeros(4)

    class Pythagoras(ClipEmbedder):
        def _encode(self, img):
            return np.array([3.0, 4.0])

    assert Zero(None, None).embed(base) == []
    assert Pythagoras(None, None).embed(base.convert("L")) == [0.6, 0.8]
