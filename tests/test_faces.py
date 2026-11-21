"""Testy twarzy (slad.features.faces) i pobierania modeli (slad.features.model_download) — bez sieci.

Prawdziwych zdjęć twarzy w testach nie ma: logikę detekcji, bazy referencyjnej i identyfikacji
sprawdzamy na atrapach detektora/rozpoznawania (embedding = kolor twarzy). Prawdziwe modele YuNet/SFace
— tylko test dymny (obraz bez twarzy → []), pomijany, gdy modeli nie pobrano.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
import time
import types
import unicodedata
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.features import faces as faces_mod  # noqa: E402
from slad.features import model_download as md  # noqa: E402
from slad.features.faces import CACHE_NAME, FaceEngine, cosine, person_name, to_bgr  # noqa: E402
from slad.features.visual import image_features  # noqa: E402
from slad.settings import ROOT, FaceSettings  # noqa: E402

DET_MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
REC_MODEL = ROOT / "models" / "face_recognition_sface_2021dec.onnx"
needs_models = pytest.mark.skipif(
    not (DET_MODEL.is_file() and REC_MODEL.is_file()), reason="brak modeli — python -m slad models"
)

RED, GREEN, BLUE, WHITE = (220, 20, 20), (20, 200, 20), (20, 20, 220), (255, 255, 255)


# ------------------------------------------------------------------ atrapy OpenCV


def face_row(x: float, y: float, w: float, h: float, score: float = 0.95) -> list[float]:
    """Wiersz w formacie YuNet: bbox, 5 punktów charakterystycznych, wynik."""
    pts = [(0.3, 0.4), (0.7, 0.4), (0.5, 0.6), (0.35, 0.8), (0.65, 0.8)]
    return [x, y, w, h, *[c for px, py in pts for c in (x + px * w, y + py * h)], score]


def center_face(img: np.ndarray) -> list[list[float]]:
    """Twarz na środku obrazu, o ile środek nie jest biały (biały obraz = „zdjęcie bez twarzy”)."""
    h, w = img.shape[:2]
    if img[h // 2, w // 2].astype(int).sum() > 700:
        return []
    return [face_row(w * 0.25, h * 0.25, w * 0.5, h * 0.5)]


class FakeDetector:
    def __init__(self, rows_fn=center_face) -> None:
        self.rows_fn = rows_fn
        self.input_size: tuple[int, int] | None = None
        self.calls = 0

    def setInputSize(self, size) -> None:  # noqa: N802 — API OpenCV
        self.input_size = tuple(size)

    def detect(self, img: np.ndarray):
        assert (img.shape[1], img.shape[0]) == self.input_size, "setInputSize musi poprzedzać detect"
        self.calls += 1
        rows = self.rows_fn(img)
        return 1, (np.asarray(rows, dtype=np.float32).reshape(-1, 15) if rows else None)


class FakeRecognizer:
    """Embedding = średni kolor (BGR) wycinka twarzy na pierwszych 3 z 128 wymiarów."""

    def __init__(self) -> None:
        self.feature_calls = 0

    def alignCrop(self, img: np.ndarray, row: np.ndarray) -> np.ndarray:  # noqa: N802 — API OpenCV
        x, y, w, h = (int(round(v)) for v in np.asarray(row, dtype=np.float32).ravel()[:4])
        crop = img[max(0, y) : y + h, max(0, x) : x + w]
        return crop if crop.size else img

    def feature(self, aligned: np.ndarray) -> np.ndarray:
        self.feature_calls += 1
        vec = np.zeros((1, 128), dtype=np.float32)
        vec[0, :3] = aligned.reshape(-1, 3).mean(axis=0) * 3.0  # celowo nieznormalizowany
        return vec


class FailingRecognizer(FakeRecognizer):
    def feature(self, aligned: np.ndarray) -> np.ndarray:
        raise RuntimeError("nie powinno być liczone — wynik jest w pamięci podręcznej")


def engine(detector=None, recognizer=None, **kw) -> FaceEngine:
    return FaceEngine(detector or FakeDetector(), recognizer or FakeRecognizer(), **kw)


def solid(color, size=(200, 200)) -> Image.Image:
    return Image.new("RGB", size, color)


@pytest.fixture
def gallery(tmp_path: Path) -> Path:
    g = tmp_path / "faces"
    (g / "Jan_Kowalski").mkdir(parents=True)
    (g / "Anna_Nowak").mkdir()
    (g / "Bez_Twarzy").mkdir()
    (g / ".ukryty").mkdir()
    solid(RED).save(g / "Jan_Kowalski" / "1.jpg", quality=95)
    solid((160, 10, 10), (300, 240)).save(g / "Jan_Kowalski" / "2.PNG")
    (g / "Jan_Kowalski" / "notatki.txt").write_text("nie obraz")
    solid(BLUE).save(g / "Anna_Nowak" / "a.webp")
    solid(WHITE).save(g / "Bez_Twarzy" / "pusty.png")
    solid(GREEN).save(g / ".ukryty" / "x.png")
    (g / ".gitkeep").write_text("")
    return g


# ------------------------------------------------------------------ pomocnicze


def test_cosine_helper() -> None:
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([1, 2], [2, 4]) == pytest.approx(1.0)
    assert cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine(np.array([3.0, 4.0]), [3, 4]) == pytest.approx(1.0)
    for a, b in (([], [1]), ([0, 0], [1, 1]), (None, [1]), ([1, 2], [1, 2, 3]), ([np.nan, 1], [1, 1])):
        assert cosine(a, b) == 0.0


def test_person_name_is_nfc_with_spaces() -> None:
    assert person_name("Jan_Kowalski") == "Jan Kowalski"
    assert person_name("  Anna__Maria_Nowak ") == "Anna Maria Nowak"
    nfd = unicodedata.normalize("NFD", "Łukasz_Żółć")
    assert person_name(nfd) == unicodedata.normalize("NFC", "Łukasz Żółć")


def test_to_bgr_accepts_common_inputs() -> None:
    rgb = Image.new("RGB", (4, 3), (255, 0, 0))
    assert to_bgr(rgb)[0, 0].tolist() == [0, 0, 255]  # PIL RGB → BGR
    rgba = Image.new("RGBA", (4, 3), (0, 0, 0, 0))
    assert to_bgr(rgba)[0, 0].tolist() == [255, 255, 255]  # przezroczystość → białe tło
    assert to_bgr(np.zeros((3, 4), np.uint8)).shape == (3, 4, 3)
    assert to_bgr(np.zeros((3, 4, 1), np.uint8)).shape == (3, 4, 3)
    assert to_bgr(np.zeros((3, 4, 4), np.uint8)).shape == (3, 4, 3)
    assert to_bgr(np.ones((3, 4, 3), np.float32)).max() == 255
    with pytest.raises(ValueError):
        to_bgr(np.zeros((3, 4, 5), np.uint8))


# ------------------------------------------------------------------ detekcja (atrapy)


def test_detect_filters_sorts_and_normalizes() -> None:
    rows = [
        face_row(10, 10, 60, 60, score=0.95),
        face_row(100, 100, 120, 130, score=0.9),  # największa → pierwsza
        face_row(300, 10, 80, 80, score=0.5),  # za niski wynik
        face_row(300, 200, 30, 30, score=0.99),  # za mała (< 40 px)
        face_row(-15, 250, 60, 60, score=0.85),  # wystaje poza kadr → bbox przycięty
    ]
    eng = engine(FakeDetector(lambda img: rows))
    faces = eng.detect(np.full((400, 500, 3), 90, np.uint8))
    assert [f.bbox for f in faces] == [(100, 100, 120, 130), (10, 10, 60, 60), (0, 250, 45, 60)]
    assert [f.score for f in faces] == [0.9, 0.95, 0.85]
    for f in faces:
        assert len(f.embedding) == 128
        assert np.linalg.norm(f.embedding) == pytest.approx(1.0, abs=1e-4)
        assert f.identity is None and f.identity_score is None  # pusta baza


def test_detect_caps_number_of_faces() -> None:
    rows = [face_row(i * 45, 0, 40 + i, 40 + i) for i in range(10)]
    faces = engine(FakeDetector(lambda img: rows), max_faces=3).detect(np.zeros((100, 600, 3), np.uint8))
    assert [f.bbox[2] for f in faces] == [49, 48, 47]


def test_detect_downscales_large_images_and_maps_back() -> None:
    det = FakeDetector(lambda img: [face_row(100, 50, 50, 50)])
    faces = engine(det, max_side=1000).detect(np.zeros((1000, 4000, 3), np.uint8))
    assert det.input_size == (1000, 250)
    assert faces[0].bbox == (400, 200, 200, 200)  # współrzędne oryginału


def test_detect_edge_cases() -> None:
    det = FakeDetector()
    eng = engine(det)
    assert eng.detect(np.zeros((30, 300, 3), np.uint8)) == []  # mniejszy niż minimalna twarz
    assert det.calls == 0
    assert eng.detect(solid(WHITE)) == []  # detektor zwraca None
    gray = eng.detect(np.full((200, 200), 60, np.uint8))
    assert len(gray) == 1 and gray[0].bbox == (50, 50, 100, 100)
    assert len(eng.detect(Image.new("RGBA", (200, 200), (200, 0, 0, 255)))) == 1


def test_failed_embedding_keeps_face_without_identity() -> None:
    class Broken(FakeRecognizer):
        def feature(self, aligned):
            raise RuntimeError("cv2.error")

    eng = engine(recognizer=Broken())
    eng.set_gallery({"Jan Kowalski": [0.0, 0.0, 1.0] + [0.0] * 125})
    faces = eng.detect(solid(RED))
    assert len(faces) == 1 and faces[0].embedding == [] and faces[0].identity is None


def test_low_quality_face_has_no_embedding() -> None:
    """Rysunek/awatar: SFace daje wektor o małej normie — twarz zostaje, ale bez embeddingu i tożsamości."""

    class Weak(FakeRecognizer):
        def feature(self, aligned):
            vec = super().feature(aligned)
            return vec / np.linalg.norm(vec) * 4.0  # norma 4 < MIN_FEATURE_NORM

    eng = engine(recognizer=Weak())
    eng.set_gallery({"Jan Kowalski": [0.0, 0.0, 1.0] + [0.0] * 125})
    faces = eng.detect(solid(RED))
    assert len(faces) == 1 and faces[0].bbox == (50, 50, 100, 100)
    assert faces[0].embedding == [] and faces[0].identity is None
    strong = engine(recognizer=Weak(), min_feature_norm=3.0)
    strong.set_gallery({"Jan Kowalski": [0.0, 0.0, 1.0] + [0.0] * 125})
    assert strong.detect(solid(RED))[0].identity == "Jan Kowalski"


def test_detect_is_serialized_between_threads() -> None:
    class ExclusiveDetector(FakeDetector):
        def __init__(self) -> None:
            super().__init__()
            self.busy = False
            self.overlaps = 0

        def detect(self, img):
            if self.busy:
                self.overlaps += 1
            self.busy = True
            time.sleep(0.002)
            self.busy = False
            return super().detect(img)

    det = ExclusiveDetector()
    eng = engine(det)
    errors: list[Exception] = []

    def work(i: int) -> None:
        try:
            for _ in range(5):
                size = (120 + 10 * i, 100 + 5 * i)  # różne rozmiary → setInputSize musi być pod blokadą
                assert len(eng.detect(np.full((size[1], size[0], 3), 50, np.uint8))) == 1
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and det.overlaps == 0 and det.calls == 40


# ------------------------------------------------------------------ identyfikacja i baza


def test_identify_with_synthetic_embeddings() -> None:
    eng = engine(match_threshold=0.363)
    e1, e2 = np.eye(128, dtype=np.float32)[:2]
    eng.set_gallery({"Jan Kowalski": [e1, e1 * 2 + 0.1 * e2], "Anna Nowak": e2})
    assert eng.people == ["Anna Nowak", "Jan Kowalski"]
    assert eng.gallery_counts == {"Anna Nowak": 1, "Jan Kowalski": 2}
    name, score = eng.identify(e1 + 0.2 * e2)
    assert name == "Jan Kowalski" and 0.9 < score <= 1.0
    assert eng.identify(e2 * 5)[0] == "Anna Nowak"
    weak = np.cos(1.3) * e1 + np.sin(1.3) * np.eye(128)[5]  # kosinus ≈ 0.27 < 0.363
    assert eng.identify(weak) == (None, None)
    assert eng.identify(np.ones(64)) == (None, None)  # inny wymiar
    assert eng.identify(None) == (None, None)
    assert eng.identify(np.zeros(128)) == (None, None)


def test_gallery_identifies_people(gallery: Path) -> None:
    eng = engine()
    counts = eng.load_gallery(gallery)
    assert counts == {"Anna Nowak": 1, "Jan Kowalski": 2}  # „Bez Twarzy” i ukryty katalog pominięte
    red = eng.detect(solid(RED, (320, 240)))
    assert red[0].identity == "Jan Kowalski" and red[0].identity_score > 0.99
    assert eng.detect(solid(BLUE))[0].identity == "Anna Nowak"
    green = eng.detect(solid(GREEN))
    assert len(green) == 1 and green[0].identity is None and green[0].identity_score is None


def test_gallery_cache_is_reused_and_updated(gallery: Path) -> None:
    rec = FakeRecognizer()
    engine(recognizer=rec).load_gallery(gallery)
    assert rec.feature_calls == 3  # 2× Jan + Anna (pusty obraz bez twarzy nie dochodzi do SFace)
    cache = json.loads((gallery / CACHE_NAME).read_text(encoding="utf-8"))
    red_sha = hashlib.sha256((gallery / "Jan_Kowalski" / "1.jpg").read_bytes()).hexdigest()
    assert red_sha in cache["files"] and cache["files"][red_sha]["file"] == "Jan_Kowalski/1.jpg"
    assert len(cache["files"]) == 4 and sum(v["embedding"] is None for v in cache["files"].values()) == 1

    cached = engine(recognizer=FailingRecognizer())  # wszystko z pamięci podręcznej
    assert cached.load_gallery(gallery) == {"Anna Nowak": 1, "Jan Kowalski": 2}
    query = engine().detect(solid(RED))[0].embedding
    assert cached.identify(query)[0] == "Jan Kowalski"

    solid(GREEN).save(gallery / "Anna_Nowak" / "a.webp")  # zmiana zdjęcia → nowy SHA → liczone od nowa
    rec2 = FakeRecognizer()
    eng2 = engine(recognizer=rec2)
    eng2.load_gallery(gallery)
    assert rec2.feature_calls == 1
    assert eng2.detect(solid(GREEN))[0].identity == "Anna Nowak"
    assert len(json.loads((gallery / CACHE_NAME).read_text(encoding="utf-8"))["files"]) == 4  # stary wpis usunięty


def test_gallery_cache_invalidated_by_model_or_corruption(gallery: Path) -> None:
    engine(model_key="model-A").load_gallery(gallery)
    rec = FakeRecognizer()
    engine(recognizer=rec, model_key="model-B").load_gallery(gallery)
    assert rec.feature_calls == 3
    (gallery / CACHE_NAME).write_text("{ to nie jest json", encoding="utf-8")
    rec = FakeRecognizer()
    assert engine(recognizer=rec, model_key="model-B").load_gallery(gallery) == {"Anna Nowak": 1, "Jan Kowalski": 2}
    assert rec.feature_calls == 3


def test_gallery_read_only_and_missing_dir(gallery: Path, tmp_path: Path, monkeypatch, caplog) -> None:
    def deny(*args, **kwargs):
        raise PermissionError("tylko do odczytu")

    monkeypatch.setattr(faces_mod.tempfile, "mkstemp", deny)
    with caplog.at_level(logging.WARNING):
        counts = engine().load_gallery(gallery)
    assert counts == {"Anna Nowak": 1, "Jan Kowalski": 2}
    assert not (gallery / CACHE_NAME).exists()
    assert "pamięci bazy twarzy" in caplog.text
    eng = engine()
    assert eng.load_gallery(tmp_path / "nie_ma") == {} and eng.people == []


# ------------------------------------------------------------------ from_settings


def test_from_settings_disabled_or_missing_models(tmp_path: Path, caplog) -> None:
    assert FaceEngine.from_settings(FaceSettings(enabled=False)) is None
    cfg = FaceSettings(detector_model=str(tmp_path / "det.onnx"), recognizer_model=str(tmp_path / "rec.onnx"))
    with caplog.at_level(logging.WARNING):
        assert FaceEngine.from_settings(cfg) is None
    assert "python -m slad models" in caplog.text


def test_from_settings_corrupted_models(tmp_path: Path, caplog) -> None:
    for name in ("det.onnx", "rec.onnx"):
        (tmp_path / name).write_bytes(b"\x00junk" * 100)
    cfg = FaceSettings(detector_model=str(tmp_path / "det.onnx"), recognizer_model=str(tmp_path / "rec.onnx"))
    with caplog.at_level(logging.WARNING):
        assert FaceEngine.from_settings(cfg) is None
    assert "python -m slad models" in caplog.text


@needs_models
def test_real_models_find_no_faces_on_synthetic_images(tmp_path: Path) -> None:
    eng = FaceEngine.from_settings(FaceSettings(gallery_dir=str(tmp_path / "pusta_baza")))
    assert eng is not None and eng.people == []
    rng = np.random.default_rng(0)
    gradient = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))
    images = [
        Image.fromarray(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)),
        Image.fromarray(gradient).convert("RGB"),
        solid(WHITE, (800, 600)),
        np.full((300, 300), 128, np.uint8),
        np.zeros((240, 320, 4), np.uint8),
        np.zeros((2400, 3000, 3), np.uint8),  # zmniejszany do detekcji
        np.zeros((20, 20, 3), np.uint8),  # mniejszy niż minimalna twarz
    ]
    for img in images:
        assert eng.detect(img) == []
    p = tmp_path / "x.png"
    Image.fromarray(rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)).save(p)
    assert image_features(p, types.SimpleNamespace(faces=eng, clip=None)).faces == []


def draw_cartoon(skin, hair, suit, bg) -> Image.Image:
    """Rysowana twarz (oczy, usta) — YuNet ją wykrywa, ale SFace daje dla wszystkich rysunków podobny wektor."""
    from PIL import ImageDraw

    img = Image.new("RGB", (600, 400), (240, 240, 245))
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box = (380, 40, 580, 320)
    w, h = x1 - x0, y1 - y0
    d.rectangle(box, fill=bg)
    d.ellipse((x0 + w * 0.08, y0 + h * 0.66, x1 - w * 0.08, y1 + h * 0.4), fill=suit)
    d.rectangle((x0 + w * 0.42, y0 + h * 0.52, x0 + w * 0.58, y0 + h * 0.7), fill=skin)
    d.ellipse((x0 + w * 0.3, y0 + h * 0.16, x0 + w * 0.7, y0 + h * 0.6), fill=skin)
    d.chord((x0 + w * 0.28, y0 + h * 0.1, x0 + w * 0.72, y0 + h * 0.46), 180, 360, fill=hair)
    for ex in (0.42, 0.58):
        d.ellipse((x0 + w * (ex - 0.025), y0 + h * 0.36, x0 + w * (ex + 0.025), y0 + h * 0.4), fill=(40, 30, 30))
    d.arc((x0 + w * 0.43, y0 + h * 0.42, x0 + w * 0.57, y0 + h * 0.52), 20, 160, fill=(120, 50, 50), width=2)
    return img


@needs_models
def test_real_models_drawn_faces_get_no_embedding(tmp_path: Path) -> None:
    """Dwa różne rysunki „osób” nie mogą dać przesłanki „ta sama twarz” (fałszywe powiązanie)."""
    eng = FaceEngine.from_settings(FaceSettings(gallery_dir=str(tmp_path / "pusta_baza")))
    assert eng is not None
    for colors in (
        ((233, 196, 160), (60, 40, 30), (40, 50, 80), (201, 214, 232)),
        ((180, 130, 100), (40, 30, 20), (20, 20, 60), (216, 239, 225)),
    ):
        faces = eng.detect(draw_cartoon(*colors))
        assert len(faces) == 1 and faces[0].score >= 0.8  # wykryta…
        assert faces[0].embedding == [] and faces[0].identity is None  # …ale bez embeddingu


@needs_models
def test_real_gallery_without_faces_is_cached(tmp_path: Path) -> None:
    g = tmp_path / "faces" / "Jan_Kowalski"
    g.mkdir(parents=True)
    solid(WHITE).save(g / "1.png")
    eng = FaceEngine.from_settings(FaceSettings(gallery_dir=str(tmp_path / "faces")))
    assert eng is not None and eng.people == []
    data = json.loads((tmp_path / "faces" / CACHE_NAME).read_text(encoding="utf-8"))
    assert "face_recognition_sface_2021dec.onnx" in data["model"]
    assert [v["embedding"] for v in data["files"].values()] == [None]


# ------------------------------------------------------------------ pobieranie modeli (httpx.MockTransport)


class FakeServer:
    """Serwer HTTP w pamięci: obsługa Range, liczenie żądań, awarie na życzenie."""

    def __init__(self, files: dict[str, bytes], ranges: bool = True, fail_once: set[str] | None = None) -> None:
        self.files = files
        self.ranges = ranges
        self.fail_once = set(fail_once or ())
        self.requests: list[tuple[str, str | None]] = []
        self.lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url, rng = str(request.url), request.headers.get("range")
        with self.lock:
            self.requests.append((url, rng))
            if rng in self.fail_once:  # pierwsza próba tego fragmentu kończy się błędem serwera
                self.fail_once.discard(rng)
                return httpx.Response(500)
        if url.endswith("/redirect"):
            return httpx.Response(302, headers={"location": url.replace("/redirect", "/det.onnx")})
        data = self.files.get(url.rsplit("/", 1)[-1])
        if data is None:
            return httpx.Response(404)
        if rng and self.ranges:
            start, end = (int(x) for x in rng.removeprefix("bytes=").split("-"))
            end = min(end, len(data) - 1)
            headers = {"content-range": f"bytes {start}-{end}/{len(data)}"}
            return httpx.Response(206, content=data[start : end + 1], headers=headers)
        return httpx.Response(200, content=data)


def mock_client(server: FakeServer) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(server), follow_redirects=True)


def use_specs(monkeypatch, payloads: dict[str, bytes], *, urls=None, fake_loader: bool = True) -> None:
    specs = tuple(
        md.ModelSpec(
            filename=name,
            kind="detector",
            urls=tuple(urls or (f"https://a.example/{name}", f"https://b.example/{name}")),
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
        )
        for name, data in payloads.items()
    )
    monkeypatch.setattr(md, "FACE_MODELS", specs)
    if fake_loader:
        monkeypatch.setattr(md, "check_loadable", lambda path, kind: None)
    monkeypatch.setattr(md.time, "sleep", lambda s: None)


def test_download_falls_back_to_mirror(tmp_path: Path, monkeypatch) -> None:
    data = bytes(range(256)) * 800  # 200 KB
    use_specs(monkeypatch, {"det.onnx": data}, urls=("https://a.example/missing/x", "https://b.example/redirect"))
    server = FakeServer({"det.onnx": data})
    with mock_client(server) as client:
        paths = md.download_face_models(tmp_path / "models", client=client)
    assert paths == [tmp_path / "models" / "det.onnx"]
    assert paths[0].read_bytes() == data
    assert not list((tmp_path / "models").glob(".tmp-*"))
    assert ("https://b.example/det.onnx", None) in server.requests  # pobranie z adresu po przekierowaniu


def test_download_ranged_parallel_with_retry(tmp_path: Path, monkeypatch) -> None:
    data = np.random.default_rng(1).integers(0, 256, 700_000, dtype=np.uint8).tobytes()
    use_specs(monkeypatch, {"det.onnx": data})
    monkeypatch.setattr(md, "PART_BYTES", 64 * 1024)
    server = FakeServer({"det.onnx": data}, fail_once={"bytes=131072-196607"})
    with mock_client(server) as client:
        [path] = md.download_face_models(tmp_path, client=client)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(data).hexdigest()
    ranged = [r for _, r in server.requests if r and r != "bytes=0-0"]
    assert len(ranged) >= 11  # 11 fragmentów + ponowiony


def test_download_without_range_support(tmp_path: Path, monkeypatch) -> None:
    data = b"\x08\x01" * 3_000_000  # 6 MB, serwer ignoruje Range
    use_specs(monkeypatch, {"det.onnx": data})
    with mock_client(FakeServer({"det.onnx": data}, ranges=False)) as client:
        [path] = md.download_face_models(tmp_path, client=client)
    assert path.stat().st_size == len(data)


def test_download_rejects_bad_content(tmp_path: Path, monkeypatch) -> None:
    good = b"\x01" * 150_000
    use_specs(monkeypatch, {"det.onnx": good})
    bad = FakeServer({"det.onnx": b"\x02" * 150_000})  # inny SHA256
    with mock_client(bad) as client, pytest.raises(RuntimeError, match="SHA256"):
        md.download_face_models(tmp_path, client=client)
    lfs = FakeServer({"det.onnx": b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n"})
    with mock_client(lfs) as client, pytest.raises(RuntimeError, match="LFS"):
        md.download_face_models(tmp_path, client=client)
    assert not list(tmp_path.glob("*.onnx")) and not list(tmp_path.glob(".tmp-*"))


def test_download_skips_valid_files_unless_forced(tmp_path: Path, monkeypatch) -> None:
    data = b"\x03" * 150_000
    use_specs(monkeypatch, {"det.onnx": data})
    (tmp_path / "det.onnx").write_bytes(data)
    server = FakeServer({"det.onnx": data})
    with mock_client(server) as client:
        assert md.download_face_models(tmp_path, client=client) == [tmp_path / "det.onnx"]
        assert server.requests == []
        md.download_face_models(tmp_path, force=True, client=client)
        assert server.requests
    (tmp_path / "det.onnx").write_bytes(b"uszkodzony")  # niepoprawny plik → pobranie ponowne
    with mock_client(FakeServer({"det.onnx": data})) as client:
        md.download_face_models(tmp_path, client=client)
    assert (tmp_path / "det.onnx").read_bytes() == data
    assert md.missing_face_models(tmp_path) == [] and md.missing_face_models(tmp_path / "x") == ["det.onnx"]


@pytest.mark.skipif(not DET_MODEL.is_file(), reason="brak modelu YuNet — python -m slad models")
def test_download_verifies_real_model_with_opencv(tmp_path: Path, monkeypatch) -> None:
    data = DET_MODEL.read_bytes()
    assert hashlib.sha256(data).hexdigest() == md.FACE_MODELS[0].sha256  # przypięty skrót
    use_specs(monkeypatch, {DET_MODEL.name: data}, fake_loader=False)  # bez atrapy: cv2 tworzy model
    with mock_client(FakeServer({DET_MODEL.name: data})) as client:
        [path] = md.download_face_models(tmp_path, client=client)
    assert md.verify_model(path, md.FACE_MODELS[0]) is None
    garbage = tmp_path / "garbage.onnx"
    garbage.write_bytes(b"\x00" * 200_000)
    spec = md.ModelSpec("garbage.onnx", "detector", (), None)
    assert "OpenCV" in (md.verify_model(garbage, spec) or "")
