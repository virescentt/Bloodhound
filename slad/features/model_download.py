"""Pobieranie modeli twarzy z OpenCV Zoo: YuNet (detektor) i SFace (rozpoznawanie).

Pliki leżą w repozytorium OpenCV Zoo w Git LFS — adresy `github.com/.../raw/...` przekierowują
na `media.githubusercontent.com`, który zwraca właściwy plik binarny (a nie wskaźnik LFS).
Najpierw próbujemy oficjalnego lustra OpenCV na Hugging Face (w testach wielokrotnie szybsze niż
serwer LFS GitHuba), potem GitHuba. Pochodzenie nie ma znaczenia dla bezpieczeństwa: każdy plik jest
weryfikowany — rozmiar, przypięty SHA256 (= oid ze wskaźnika LFS w opencv_zoo = X-Linked-ETag lustra)
i to, czy OpenCV potrafi z niego utworzyć model. Zapis atomowy (plik tymczasowy + rename).

Duże pliki pobieramy równolegle fragmentami (HTTP Range) — wolne łącza (np. Wi-Fi na hackathonie)
dławią pojedyncze połączenie; gdy serwer nie obsługuje Range, pobieramy jednym strumieniem.

Użycie:  python -m slad models   (CLI)   albo   python -m slad.features.model_download [katalog] [--force]
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

MIN_BYTES = 100_000  # wskaźnik Git LFS / strona błędu mają kilkaset bajtów
MAX_BYTES = 200_000_000  # bezpiecznik na wypadek błędnej odpowiedzi serwera
PART_BYTES = 1 << 20  # fragment pobierania równoległego
PARALLEL = 8  # liczba równoległych połączeń
PART_RETRIES = 4
_LFS_POINTER = b"version https://git-lfs"
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
_HF = "https://huggingface.co/opencv"


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    kind: str  # "detector" | "recognizer"
    urls: tuple[str, ...]
    sha256: str | None  # przypięty skrót pliku (None = tylko kontrola rozmiaru i wczytania)
    size: int | None = None


# SHA256 = oid ze wskaźnika Git LFS w opencv_zoo (main) = X-Linked-ETag lustra Hugging Face.
FACE_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        filename="face_detection_yunet_2023mar.onnx",
        kind="detector",
        urls=(
            f"{_HF}/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
            f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ),
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        size=232_589,
    ),
    ModelSpec(
        filename="face_recognition_sface_2021dec.onnx",
        kind="recognizer",
        urls=(
            f"{_HF}/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
            f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        ),
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        size=38_696_353,
    ),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def quiet_opencv() -> Iterator[None]:
    """Wycisza ostrzeżenia OpenCV przy tworzeniu sieci (np. „Targets are not supported by the new graph engine”)."""
    import cv2

    api = getattr(getattr(cv2, "utils", None), "logging", None)
    previous = None
    try:
        if api is not None:
            previous = api.getLogLevel()
            api.setLogLevel(api.LOG_LEVEL_ERROR)
    except Exception:  # noqa: BLE001 — wyciszenie jest tylko kosmetyką
        api = None
    try:
        yield
    finally:
        if api is not None and previous is not None:
            api.setLogLevel(previous)


def check_loadable(path: Path, kind: str) -> None:
    """Sprawdza, czy OpenCV tworzy model z pliku (cv2 rozpoznaje format po rozszerzeniu .onnx)."""
    import cv2

    try:
        with quiet_opencv():
            if kind == "detector":
                cv2.FaceDetectorYN.create(str(path), "", (320, 320))
            else:
                cv2.FaceRecognizerSF.create(str(path), "")
    except cv2.error as e:
        raise RuntimeError(f"OpenCV nie wczytuje modelu {path.name}: {e}") from e


def _make_readable(path: Path) -> None:
    """Modele nie są tajne: 0644, żeby czytał je też inny użytkownik (np. kontener Dockera)."""
    try:
        mode = path.stat().st_mode
        if mode & 0o044 != 0o044:
            os.chmod(path, mode | 0o044)
    except OSError:
        pass


def verify_model(path: Path, spec: ModelSpec, *, digest: str | None = None) -> str | None:
    """Zwraca opis problemu z plikiem modelu albo None, gdy plik jest poprawny."""
    if not path.is_file():
        return "brak pliku"
    size = path.stat().st_size
    if size < MIN_BYTES:
        with open(path, "rb") as f:
            if f.read(len(_LFS_POINTER)) == _LFS_POINTER:
                return "wskaźnik Git LFS zamiast pliku modelu"
        return f"plik za mały ({size} B)"
    digest = digest or sha256_file(path)
    if spec.sha256 and digest != spec.sha256:
        return f"niezgodny SHA256 ({digest[:12]}…, oczekiwano {spec.sha256[:12]}…)"
    try:
        check_loadable(path, spec.kind)
    except Exception as e:  # noqa: BLE001
        return str(e)
    return None


def missing_face_models(dest_dir: Path) -> list[str]:
    """Nazwy plików modeli, których brakuje w katalogu (szybka kontrola, bez weryfikacji SHA)."""
    return [s.filename for s in FACE_MODELS if not (Path(dest_dir) / s.filename).is_file()]


# ------------------------------------------------------------------ pobieranie


def _probe(client, url: str) -> tuple[str, int | None]:
    """Adres końcowy (po przekierowaniach) i rozmiar pliku, jeśli serwer obsługuje Range (inaczej None)."""
    with client.stream("GET", url, headers={"Range": "bytes=0-0"}) as r:  # treści nie czytamy
        r.raise_for_status()
        final = str(r.url)
        if r.status_code == 206:
            total = r.headers.get("content-range", "").rpartition("/")[2]
            if total.isdigit():
                return final, int(total)
    return final, None


def _stream_to(client, url: str, f) -> int:
    total = 0
    with client.stream("GET", url) as r:
        r.raise_for_status()
        for chunk in r.iter_bytes(chunk_size=1 << 16):
            total += len(chunk)
            if total > MAX_BYTES:
                raise RuntimeError(f"odpowiedź większa niż {MAX_BYTES} B")
            f.write(chunk)
    return total


def _ranged_to(client, url: str, path: Path, total: int, label: str = "") -> None:
    with open(path, "r+b") as f:
        f.truncate(total)

    def part(start: int) -> None:
        """Pobiera bajty [start, end]; zerwane połączenie wznawia od ostatniego odebranego bajtu.

        Poddaje się dopiero po PART_RETRIES kolejnych próbach bez żadnego postępu.
        """
        end = min(total, start + PART_BYTES) - 1
        pos, stalls = start, 0
        while pos <= end:
            before = pos
            try:
                with client.stream("GET", url, headers={"Range": f"bytes={pos}-{end}"}) as r:
                    content_range = r.headers.get("content-range", "")
                    if r.status_code != 206 or not content_range.startswith(f"bytes {pos}-"):
                        raise RuntimeError(f"HTTP {r.status_code}, Content-Range {content_range!r}")
                    with open(path, "r+b") as f:
                        f.seek(pos)
                        for chunk in r.iter_bytes(chunk_size=1 << 16):
                            if pos + len(chunk) > end + 1:
                                raise RuntimeError("serwer zwrócił więcej danych niż fragment")
                            f.write(chunk)
                            pos += len(chunk)
                if pos <= end:
                    raise RuntimeError(f"niepełny fragment ({pos - start}/{end - start + 1} B)")
            except Exception as e:  # noqa: BLE001 — ponawiamy (od miejsca przerwania)
                stalls = 0 if pos > before else stalls + 1
                if stalls >= PART_RETRIES:
                    raise RuntimeError(f"nie udało się pobrać fragmentu {start}-{end}: {e}") from e
                time.sleep(0.5 * (stalls + 1))

    with ThreadPoolExecutor(PARALLEL) as ex:
        futures = [ex.submit(part, start) for start in range(0, total, PART_BYTES)]
        try:
            reported = 0
            for i, fut in enumerate(futures, 1):
                fut.result()
                pct = i * 100 // len(futures)
                if pct >= reported + 10:  # postęp co ~10% (wolne łącza: pobieranie trwa minuty)
                    reported = pct - pct % 10
                    log.info("%s: %d%%", label or path.name, pct)
        except BaseException:
            for fut in futures:
                fut.cancel()
            raise


def _download_one(client, url: str, spec: ModelSpec, dest_dir: Path) -> Path:
    target = dest_dir / spec.filename
    # sufiks .onnx jest potrzebny, żeby cv2 rozpoznał format przy weryfikacji pliku tymczasowego
    fd, tmp_name = tempfile.mkstemp(dir=dest_dir, prefix=".tmp-", suffix=".onnx")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        t0 = time.monotonic()
        final, total = _probe(client, url)
        if total is not None and total > MAX_BYTES:
            raise RuntimeError(f"plik większy niż {MAX_BYTES} B")
        if total is not None and total > 2 * PART_BYTES:
            _ranged_to(client, final, tmp, total, spec.filename)
        else:
            with open(tmp, "wb") as f:
                total = _stream_to(client, final, f)
        digest = sha256_file(tmp)
        problem = verify_model(tmp, spec, digest=digest)
        if problem:
            raise RuntimeError(problem)
        _make_readable(tmp)  # mkstemp tworzy pliki 0600
        os.replace(tmp, target)
        log.info("Pobrano %s (%d B w %.0f s, sha256 %s)", spec.filename, total, time.monotonic() - t0, digest)
        return target
    finally:
        tmp.unlink(missing_ok=True)


def download_face_models(dest_dir: Path, force: bool = False, *, client=None) -> list[Path]:
    """Pobiera YuNet i SFace do `dest_dir` (pomija poprawne pliki, chyba że force=True).

    Zwraca ścieżki obu modeli. Gdy któregoś nie udało się pobrać z żadnego adresu — RuntimeError
    (poprawnie pobrane pliki zostają na dysku). `client` (httpx.Client) — do testów bez sieci.
    """
    import httpx

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    own_client = client is None
    if own_client:
        client = httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(60.0, connect=15.0),
            limits=httpx.Limits(max_connections=PARALLEL * 2, max_keepalive_connections=PARALLEL),
        )
    paths: list[Path] = []
    failures: list[str] = []
    try:
        for spec in FACE_MODELS:
            target = dest_dir / spec.filename
            if not force and target.exists():
                problem = verify_model(target, spec)
                if problem is None:
                    log.info("Model %s już jest — pomijam", spec.filename)
                    _make_readable(target)
                    paths.append(target)
                    continue
                log.warning("Model %s jest niepoprawny (%s) — pobieram ponownie", spec.filename, problem)
            errors: list[str] = []
            for url in spec.urls:
                try:
                    log.info("Pobieram %s z %s", spec.filename, url)
                    paths.append(_download_one(client, url, spec, dest_dir))
                    break
                except Exception as e:  # noqa: BLE001 — próbujemy kolejnego adresu
                    log.warning("Nie udało się pobrać %s z %s: %s", spec.filename, url, e)
                    errors.append(f"{url}: {e}")
            else:
                failures.append(f"{spec.filename} — " + "; ".join(errors))
    finally:
        if own_client:
            client.close()
    if failures:
        raise RuntimeError("Nie udało się pobrać modeli twarzy: " + " | ".join(failures))
    return paths


def main(argv: list[str] | None = None) -> int:
    import sys

    from slad.settings import ROOT

    args = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in args
    rest = [a for a in args if a != "--force"]
    dest = Path(rest[0]) if rest else ROOT / "models"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        for p in download_face_models(dest, force=force):
            log.info("OK  %s", p)
    except RuntimeError as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
