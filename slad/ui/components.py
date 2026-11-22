"""Czyste funkcje pomocnicze interfejsu (bez Streamlit): modele → tabele pandas, kolory, etykiety.

Wszystko tutaj da się przetestować bez uruchomionego Streamlit. Teksty widoczne w UI — po polsku.
Treści pochodzące z materiałów (tekst reklam, adresy, OCR) są niezaufane: w UI pokazujemy je jako
zwykły tekst (komórki tabel, ``st.text``) albo po ``md_escape`` — nigdy jako aktywny Markdown/HTML.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

from slad.models import (
    Assignment,
    Campaign,
    CorpusStats,
    Evidence,
    FeatureSet,
    ImageFeatures,
    InfraFeatures,
    Link,
    Material,
    SourceKind,
    TextFeatures,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- kolory i etykiety

# Paleta kategoryczna — stała kolejność slotów (sprawdzona pod kątem daltonizmu, jasne tło).
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # niebieski
    "#eb6834",  # pomarańczowy
    "#1baf7a",  # morski
    "#eda100",  # żółty
    "#e87ba4",  # magenta
    "#008300",  # zielony
    "#6250d6",  # fioletowy
    "#e34948",  # czerwony
)
# Kształty węzłów — kodowanie złożone (kolor × kształt) dla kampanii powyżej 8.
SHAPES: tuple[str, ...] = ("dot", "square", "triangle", "triangleDown", "star", "hexagon")
SHAPE_GLYPHS: dict[str, str] = {
    "dot": "●",
    "square": "■",
    "triangle": "▲",
    "triangleDown": "▼",
    "star": "★",
    "hexagon": "⬢",
    "diamond": "◆",
}

INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#898781"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
UNASSIGNED_COLOR = BASELINE  # materiał bez kampanii — szary, „cofa się” w tło

# Grupy przesłanek (Evidence.group) — kolejność wyświetlania = kolejność slotów palety.
GROUP_ORDER: tuple[str, ...] = ("infrastruktura", "tekst", "obraz", "osoby", "narracja")
GROUP_COLORS: dict[str, str] = {g: CATEGORICAL[i] for i, g in enumerate(GROUP_ORDER)}
GROUP_ICONS: dict[str, str] = {
    "infrastruktura": "🌐",
    "tekst": "📝",
    "obraz": "🖼️",
    "osoby": "👤",
    "narracja": "🧭",
}

RELATIONS: tuple[str, ...] = ("duplikat", "wariant", "powiązany")
# Relacja jest uporządkowana (siła) → jednobarwna skala szarości + wzór linii jako drugi kanał.
RELATION_COLORS: dict[str, str] = {"duplikat": INK_2, "wariant": INK_MUTED, "powiązany": BASELINE}
RELATION_DASHES: dict[str, bool] = {"duplikat": False, "wariant": False, "powiązany": True}
RELATION_ICONS: dict[str, str] = {"duplikat": "🟰", "wariant": "≈", "powiązany": "🔗"}

# Status przypisania nowego materiału: (etykieta, kolor st.badge, ikona)
STATUS_VIEW: dict[str, tuple[str, str, str]] = {
    "przypisany": ("Przypisany do znanej kampanii", "blue", ":material/link:"),
    "nowa_kampania": ("Potencjalnie nowa kampania", "orange", ":material/new_releases:"),
    "brak_powiązań": ("Brak powiązań z bazą", "gray", ":material/link_off:"),
}

# Rodzaje przesłanek (klucze wag z config/settings.yaml → link.weights): (grupa, opis)
KIND_INFO: dict[str, tuple[str, str]] = {
    "tracker_id": ("infrastruktura", "wspólny identyfikator śledzący (GA, GTM, Pixel, Metrica, TikTok)"),
    "crypto_wallet": ("infrastruktura", "ten sam portfel kryptowalut"),
    "phone": ("infrastruktura", "ten sam numer telefonu"),
    "email": ("infrastruktura", "ten sam adres e-mail"),
    "telegram": ("infrastruktura", "ten sam kontakt w Telegramie"),
    "whatsapp": ("infrastruktura", "ten sam numer WhatsApp"),
    "tls_fingerprint": ("infrastruktura", "ten sam certyfikat TLS"),
    "domain": ("infrastruktura", "ta sama domena (eTLD+1)"),
    "favicon": ("infrastruktura", "ta sama favicona (hash mmh3)"),
    "form_action": ("infrastruktura", "ten sam adres docelowy formularza"),
    "form_fields": ("infrastruktura", "ten sam nietypowy zestaw pól formularza"),
    "redirect_domain": ("infrastruktura", "wspólna domena pośrednicząca w przekierowaniu"),
    "cert_domain": ("infrastruktura", "domena w tym samym certyfikacie TLS"),
    "host": ("infrastruktura", "ten sam host"),
    "dom_simhash": ("infrastruktura", "ten sam szablon strony (simhash DOM)"),
    "ip": ("infrastruktura", "ten sam adres IP"),
    "title": ("infrastruktura", "ten sam tytuł strony"),
    "registrar_created": ("infrastruktura", "ten sam rejestrator i data rejestracji (±3 dni)"),
    "nameserver": ("infrastruktura", "te same serwery nazw (DNS)"),
    "asn": ("infrastruktura", "ten sam system autonomiczny (ASN)"),
    "text_minhash": ("tekst", "niemal identyczny tekst (MinHash)"),
    "text_tfidf": ("tekst", "podobna treść (TF-IDF)"),
    "keyword": ("tekst", "wspólne słowa kluczowe"),
    "image_phash": ("obraz", "ta sama lub przerobiona grafika (pHash)"),
    "image_clip": ("obraz", "podobna grafika (CLIP)"),
    "face": ("osoby", "ta sama twarz"),
    "person": ("osoby", "ta sama osoba (nazwisko / tożsamość z bazy)"),
    "brand": ("narracja", "ta sama podszywana marka lub instytucja"),
    "narrative": ("narracja", "ta sama narracja"),
}

IOC_LABELS: dict[str, str] = {
    "phone": "telefon",
    "email": "e-mail",
    "url": "adres URL",
    "domain": "domena",
    "crypto_btc": "portfel BTC",
    "crypto_eth": "portfel ETH",
    "crypto_trx": "portfel TRX (Tron)",
    "telegram": "Telegram",
    "whatsapp": "WhatsApp",
    "iban": "IBAN",
}

COVERAGE_GROUPS: tuple[tuple[str, str], ...] = (
    ("tekst", "📝 Tekst"),
    ("obraz", "🖼️ Obraz"),
    ("twarze", "🙂 Twarze"),
    ("infrastruktura", "🌐 Infrastruktura"),
)

EVIDENCE_COLUMNS: list[str] = ["Grupa", "Przesłanka", "Wartość", "Waga"]

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def tint(hex_color: str, alpha: float = 0.2, base: str = "#ffffff") -> str:
    """Kolor zmieszany z tłem (jasny odcień do tła komórek tabel)."""
    if not _HEX.match(hex_color or ""):
        return base
    c = [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(base[i : i + 2], 16) for i in (1, 3, 5)]
    mixed = [round(b[k] + (c[k] - b[k]) * alpha) for k in range(3)]
    return "#" + "".join(f"{v:02x}" for v in mixed)


def is_hex_color(value: str) -> bool:
    return bool(_HEX.match(value or ""))


def group_label(group: str) -> str:
    g = (group or "").strip() or "inne"
    return f"{GROUP_ICONS.get(g, '•')} {g}"


def group_from_label(label: str) -> str | None:
    for g in GROUP_ICONS:
        if label == group_label(g):
            return g
    return None


def kind_label(kind: str) -> str:
    info = KIND_INFO.get(kind)
    return info[1] if info else kind


def kind_group(kind: str) -> str:
    info = KIND_INFO.get(kind)
    return info[0] if info else "—"


def relation_value(rel: Any) -> str:
    return getattr(rel, "value", None) or str(rel)


def campaign_styles(campaigns: Iterable[Campaign]) -> dict[str, tuple[str, str]]:
    """Kampania → (kolor, kształt). Stałe przypisanie po identyfikatorze — filtr nie przemalowuje kampanii."""
    ids = sorted({c.id for c in campaigns})
    out: dict[str, tuple[str, str]] = {}
    for i, cid in enumerate(ids):
        out[cid] = (CATEGORICAL[i % len(CATEGORICAL)], SHAPES[(i // len(CATEGORICAL)) % len(SHAPES)])
    return out


# ---------------------------------------------------------------- tekst i bezpieczeństwo

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:=])")
_MD_SAFE = re.compile(r"([\\`\[\]()<>!#|$~=])")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


_ZWSP = "​"


def _break_autolinks(s: str) -> str:
    """GFM zamienia adresy i e-maile w linki także po escapowaniu — rozbijamy je niewidoczną spacją."""
    return s.replace("://", f":{_ZWSP}//").replace("www.", f"www{_ZWSP}.").replace("@", f"@{_ZWSP}")


def md_escape(text: Any) -> str:
    """Niezaufany tekst → bezpieczny Markdown (bez linków, obrazów, HTML, formatowania)."""
    s = _break_autolinks(_CTRL.sub(" ", str(text if text is not None else "")))
    s = _MD_SPECIAL.sub(r"\\\1", s)
    return s.replace("\n", "  \n")


def md_safe(text: Any) -> str:
    """Nasz tekst (np. uzasadnienie) z wartościami z materiałów: zostawia pogrubienia,
    neutralizuje linki, obrazy, HTML i automatyczne linkowanie adresów."""
    s = _break_autolinks(_CTRL.sub(" ", str(text if text is not None else "")))
    s = _MD_SAFE.sub(r"\\\1", s)
    return s.replace("\n", "  \n")


_URL_PARTS = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*://)?([^/?#]*)(.*)$")


def defang(url: Any) -> str:
    """Adres „rozbrojony” (hxxps://example[.]com/…) — czytelny dla analityka, ale nieklikalny."""
    s = clean_text(url)
    if not s:
        return ""
    s = re.sub(r"(?i)^http", "hxxp", s)
    m = _URL_PARTS.match(s)
    if not m:
        return s
    scheme, host, rest = m.groups()
    return (scheme or "") + host.replace(".", "[.]") + rest


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", _CTRL.sub(" ", str(text or ""))).strip()


def shorten(text: Any, n: int = 80) -> str:
    s = clean_text(text)
    return s if len(s) <= n else s[: max(1, n - 1)].rstrip() + "…"


def join_values(values: Iterable[Any], limit: int = 12, sep: str = ", ") -> str:
    vals = [clean_text(v) for v in values if v is not None and clean_text(v)]
    if not vals:
        return ""
    head = sep.join(vals[:limit])
    return head + (f" … (+{len(vals) - limit})" if len(vals) > limit else "")


def fmt_dt(value: datetime | None, with_time: bool = False) -> str:
    if value is None:
        return ""
    try:
        return value.strftime("%Y-%m-%d %H:%M" if with_time else "%Y-%m-%d")
    except (AttributeError, ValueError):
        return str(value)


def fmt_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def host_of(url: str | None) -> str | None:
    if not url:
        return None
    u = url.strip()
    if "://" not in u:
        u = "http://" + u
    try:
        return urlsplit(u).hostname
    except ValueError:
        return None


_SAFE_FILE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str, max_len: int = 60) -> str:
    stem = _SAFE_FILE.sub("_", name or "").strip("._")[:max_len]
    return stem or "plik"


# ---------------------------------------------------------------- dane katalogu roboczego

WS_FILES: tuple[str, ...] = ("materials.jsonl", "links.jsonl", "campaigns.json", "stats.json", "run.json")
RUN_FILE = "run.json"  # podsumowanie ostatniego przetwarzania (zapisuje slad.pipeline)


def workspace_signature(root: str | Path) -> tuple:
    """Klucz pamięci podręcznej: czasy modyfikacji i rozmiary plików katalogu roboczego."""
    root = Path(root)
    sig: list[tuple] = [("root", str(root.resolve()))]
    for name in WS_FILES:
        try:
            st = (root / name).stat()
            sig.append((name, st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append((name, 0, 0))
    fdir = root / "features"
    try:
        st = fdir.stat()
        sig.append(("features", st.st_mtime_ns, sum(1 for _ in fdir.glob("*.json"))))
    except OSError:
        sig.append(("features", 0, 0))
    return tuple(sig)


def is_workspace(path: Path) -> bool:
    return path.is_dir() and any((path / n).exists() for n in (*WS_FILES, "features"))


def list_workspaces(base: Path, extra: Iterable[Path] = ()) -> list[Path]:
    """Kandydaci na katalog roboczy: podane jawnie (np. z ustawień) + podkatalogi ``workspace/``."""
    seen: set[str] = set()
    out: list[Path] = []

    def add(p: Path) -> None:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p.resolve())

    for p in extra:
        add(Path(p))
    try:
        if base.is_dir():
            for d in sorted(base.iterdir()):
                if d.is_dir() and not d.name.startswith("."):
                    add(d)
    except OSError as e:
        log.warning("Nie można odczytać katalogu %s: %s", base, e)
    return out


def preferred_workspace(options: Sequence[Path], explicit: bool = False) -> Path | None:
    """Katalog wybierany na starcie: jawnie wskazany (SLAD_WORKSPACE) albo z ustawień, gdy ma dane;
    w przeciwnym razie pierwszy katalog z materiałami (np. workspace/demo po `python -m slad demo`)."""
    if not options:
        return None
    first = options[0]
    if explicit or (first / "materials.jsonl").exists():
        return first
    return next((p for p in options if (p / "materials.jsonl").exists()), first)


def workspace_label(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            pass
    return str(path)


@dataclass
class Bundle:
    """Wczytany katalog roboczy + indeksy pomocnicze dla UI."""

    root: Path
    materials: dict[str, Material] = field(default_factory=dict)
    features: dict[str, FeatureSet] = field(default_factory=dict)
    links: list[Link] = field(default_factory=list)
    campaigns: list[Campaign] = field(default_factory=list)
    stats: CorpusStats = field(default_factory=CorpusStats)
    errors: list[str] = field(default_factory=list)
    run_info: dict[str, Any] | None = None
    campaign_of: dict[str, str] = field(init=False, default_factory=dict)
    campaign_by_id: dict[str, Campaign] = field(init=False, default_factory=dict)
    links_by_material: dict[str, list[Link]] = field(init=False, default_factory=dict)
    styles: dict[str, tuple[str, str]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.reindex()

    def reindex(self) -> None:
        self.campaign_by_id = {c.id: c for c in self.campaigns}
        self.campaign_of = {}
        for c in self.campaigns:
            for mid in c.members:
                self.campaign_of.setdefault(mid, c.id)
        by_mat: dict[str, list[Link]] = defaultdict(list)
        for lk in self.links:
            by_mat[lk.a].append(lk)
            if lk.b != lk.a:
                by_mat[lk.b].append(lk)
        self.links_by_material = {k: sorted(v, key=lambda x: -x.score) for k, v in by_mat.items()}
        self.styles = campaign_styles(self.campaigns)

    def label(self, mid: str, max_len: int = 70) -> str:
        return material_label(mid, self.materials.get(mid), self.features.get(mid), max_len=max_len)

    def campaign_label(self, cid: str | None) -> str:
        if not cid:
            return "bez kampanii"
        c = self.campaign_by_id.get(cid)
        return f"{cid} · {shorten(c.name, 60)}" if c else cid


def _read_jsonl(path: Path, model: Any, errors: list[str]) -> list[Any]:
    out: list[Any] = []
    if not path.exists():
        return out
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        errors.append(f"{path.name}: {e}")
        return out
    bad = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            out.append(model.model_validate_json(line))
        except Exception:  # noqa: BLE001 — uszkodzony wiersz nie blokuje UI
            bad += 1
    if bad:
        errors.append(f"{path.name}: pominięto {bad} uszkodzonych wierszy")
    return out


def load_bundle(root: str | Path) -> Bundle:
    """Wczytuje katalog roboczy odpornie na błędy: uszkodzony plik → wpis w ``errors``, nie wyjątek."""
    import json

    root = Path(root)
    errors: list[str] = []
    materials = _read_jsonl(root / "materials.jsonl", Material, errors)
    links = _read_jsonl(root / "links.jsonl", Link, errors)

    campaigns: list[Campaign] = []
    cfile = root / "campaigns.json"
    if cfile.exists():
        try:
            raw = json.loads(cfile.read_text(encoding="utf-8"))
            for item in raw if isinstance(raw, list) else []:
                try:
                    campaigns.append(Campaign.model_validate(item))
                except Exception:  # noqa: BLE001
                    errors.append("campaigns.json: pominięto uszkodzoną kampanię")
        except Exception as e:  # noqa: BLE001
            errors.append(f"campaigns.json: {e}")

    stats = CorpusStats()
    sfile = root / "stats.json"
    if sfile.exists():
        try:
            stats = CorpusStats.model_validate_json(sfile.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            errors.append(f"stats.json: {e}")

    run_info: dict[str, Any] | None = None
    rfile = root / RUN_FILE
    if rfile.exists():
        try:
            raw_run = json.loads(rfile.read_text(encoding="utf-8"))
            run_info = raw_run if isinstance(raw_run, dict) else None
        except Exception as e:  # noqa: BLE001
            errors.append(f"{RUN_FILE}: {e}")

    features: dict[str, FeatureSet] = {}
    if materials:
        from slad.store import Workspace

        ws = Workspace(root)
        bad: list[str] = []
        for m in materials:
            try:
                fs = ws.features(m.id)
            except Exception:  # noqa: BLE001
                bad.append(m.id)
                continue
            if fs is not None:
                features[m.id] = fs
        if bad:
            errors.append(f"cechy: nie udało się odczytać {len(bad)} plików (np. {bad[0]})")

    return Bundle(
        root=root,
        materials={m.id: m for m in materials},
        features=features,
        links=links,
        campaigns=campaigns,
        stats=stats,
        errors=errors,
        run_info=run_info,
    )


def run_summary(info: Mapping[str, Any] | None) -> tuple[str, pd.DataFrame, list[str]]:
    """run.json (podsumowanie process_dataset) → (zdanie, tabela szczegółów, przykładowe błędy)."""
    if not info:
        return "", pd.DataFrame(columns=["Cecha", "Wartość"]), []

    def num(key: str) -> Any:
        v = info.get(key)
        return v if isinstance(v, int | float) and not isinstance(v, bool) else None

    parts: list[str] = []
    finished = str(info.get("finished_at") or "")
    if finished:
        parts.append(finished.replace("T", " ")[:16] + " UTC")
    dataset = str(info.get("dataset") or "")
    if dataset:
        parts.append(f"zbiór: {Path(dataset).name}")
    if num("materials") is not None:
        line = f"{num('materials')} materiałów"
        if num("time_s") is not None:
            line += f" w {num('time_s'):.0f} s"
        parts.append(line)
    if num("errors"):
        parts.append(f"materiały z błędami: {num('errors')}")
    sentence = "Ostatnie przetwarzanie: " + " · ".join(parts) if parts else ""

    settings = info.get("settings") if isinstance(info.get("settings"), Mapping) else {}

    def yn(v: Any) -> str:
        return "tak" if v else "nie"

    times = f"{info.get('extract_s', '—')} / {info.get('link_s', '—')} / {info.get('time_s', '—')}"
    rows = [
        ("Zbiór", dataset),
        ("Materiały", num("materials")),
        ("Powtórzone identyfikatory (pominięte)", num("duplicate_ids")),
        ("Cechy wyznaczone teraz / z pamięci podręcznej", f"{info.get('extracted', '—')} / {info.get('cached', '—')}"),
        ("Materiały wzbogacone przez LLM", num("llm_used")),
        ("Powiązania", num("links")),
        ("Kampanie / materiały w kampaniach", f"{info.get('campaigns', '—')} / {info.get('clustered', '—')}"),
        ("Największa kampania", num("largest_campaign")),
        ("Czas: ekstrakcja / łączenie / razem [s]", times),
        ("Sieć / OCR / LLM / twarze", " / ".join(yn(settings.get(k)) for k in ("network", "ocr", "llm", "faces"))),
        ("Model LLM", settings.get("llm_model") or ""),
    ]
    frame = pd.DataFrame(
        [{"Cecha": k, "Wartość": str(v)} for k, v in rows if v not in (None, "")], columns=["Cecha", "Wartość"]
    )
    samples = [clean_text(s) for s in info.get("error_samples") or [] if clean_text(s)]
    if info.get("link_error"):
        samples.insert(0, "łączenie: " + clean_text(info["link_error"]))
    return sentence, frame, samples


# ---------------------------------------------------------------- etykiety materiałów


def text_snippet(m: Material | None, fs: FeatureSet | None = None, n: int = 160) -> str:
    candidates: list[str | None] = []
    if m is not None:
        candidates.append(m.text)
    if fs is not None:
        t = fs.text
        candidates += [t.input_text, t.ocr_text, t.page_text, fs.infra.title]
    for c in candidates:
        if c and clean_text(c):
            return shorten(c, n)
    return ""


def material_label(mid: str, m: Material | None = None, fs: FeatureSet | None = None, max_len: int = 70) -> str:
    hint: str | None = None
    if m is not None:
        hint = m.advertiser or host_of(m.url)
    if not hint and fs is not None:
        hint = next(iter(fs.infra.domains), None) or next(iter(fs.text.persons), None)
    if not hint and m is not None and m.image_paths:
        hint = Path(m.image_paths[0]).name
    if not hint:
        hint = text_snippet(m, fs, 40) or None
    label = f"{mid} · {clean_text(hint)}" if hint else mid
    return shorten(label, max_len)


def first_image(m: Material | None, fs: FeatureSet | None = None) -> Path | None:
    paths: list[str] = []
    if m is not None:
        paths += m.image_paths
    if fs is not None:
        paths += [img.path for img in fs.images]
    for p in paths:
        try:
            pp = Path(p)
            if pp.is_file():
                return pp
        except (OSError, ValueError):
            continue
    return None


def thumbnail_bytes(path: str | Path, max_side: int = 360) -> bytes | None:
    """Miniatura PNG/JPEG obrazu; nieczytelny plik → None."""
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            im.draft("RGB", (max_side, max_side))
            im = ImageOps.exif_transpose(im).convert("RGB")  # zdjęcia z telefonu: orientacja z EXIF
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
    except Exception as e:  # noqa: BLE001
        log.debug("Miniatura %s: %s", path, e)
        return None


def faces_overlay(path: str | Path, faces: Sequence[Any], max_side: int = 640) -> bytes | None:
    """Obraz z zaznaczonymi twarzami (ramka + tożsamość), PNG; brak obrazu → None."""
    try:
        from PIL import Image, ImageDraw, ImageOps

        with Image.open(path) as im0:  # ramki twarzy są w układzie obrazu po obrocie z EXIF (jak w detekcji)
            im = ImageOps.exif_transpose(im0).convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h)) if max(w, h) else 1.0
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        draw = ImageDraw.Draw(im)
        for f in faces:
            x, y, fw, fh = (int(v * scale) for v in f.bbox)
            color = "#1baf7a" if f.identity else "#eb6834"
            draw.rectangle([x, y, x + fw, y + fh], outline=color, width=3)
            label = f.identity or "nieznana twarz"
            if f.identity_score is not None:
                label += f" ({f.identity_score:.2f})"
            tw = int(draw.textlength(label)) + 8
            ty = max(0, y - 16)
            draw.rectangle([x, ty, x + tw, ty + 15], fill=color)
            draw.text((x + 4, ty + 2), label, fill="#ffffff")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:  # noqa: BLE001
        log.debug("Twarze na obrazie %s: %s", path, e)
        return None


# ---------------------------------------------------------------- przesłanki (Evidence)


def evidence_value(ev: Evidence) -> str:
    parts: list[str] = []
    if ev.value not in (None, ""):
        parts.append(shorten(ev.value, 120))
    if ev.similarity is not None:
        parts.append(f"podobieństwo {ev.similarity:.0%}")
    return " · ".join(parts) or "—"


def evidence_frame(evidence: Iterable[Evidence], limit: int | None = None) -> pd.DataFrame:
    """Przesłanki → tabela Grupa | Przesłanka | Wartość | Waga (malejąco po wadze)."""
    rows = sorted(evidence, key=lambda e: -(e.weight or 0.0))
    if limit:
        rows = rows[:limit]
    data = [
        {
            "Grupa": group_label(e.group),
            "Przesłanka": clean_text(e.detail) or kind_label(e.kind),
            "Wartość": evidence_value(e),
            "Waga": round(float(e.weight or 0.0), 2),
        }
        for e in rows
    ]
    return pd.DataFrame(data, columns=EVIDENCE_COLUMNS)


EVIDENCE_CSS = """
.slad-ev { width: 100%; border-collapse: collapse; font-size: 0.9rem; color: #0b0b0b; margin: 2px 0 10px; }
.slad-ev th { text-align: left; font-weight: 600; color: #52514e; padding: 6px 8px; border-bottom: 1px solid #e1e0d9; }
.slad-ev td { padding: 6px 8px; border-bottom: 1px solid #f0efec; vertical-align: top; }
.slad-ev td.g { white-space: nowrap; font-weight: 600; border-radius: 4px; }
.slad-ev td.v { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; word-break: break-all; }
.slad-ev td.t { word-break: break-word; }
.slad-sum { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px 18px; margin: 4px 0 8px; }
.slad-sum .box { border: 1px solid rgba(11,11,11,0.10); border-radius: 8px; padding: 8px 10px; }
.slad-sum .h { font-weight: 600; margin-bottom: 4px; }
.slad-sum .it { display: flex; gap: 8px; font-size: 0.88rem; padding: 2px 0; }
.slad-sum .n { min-width: 26px; text-align: right; color: #52514e; font-variant-numeric: tabular-nums; }
.slad-sum .x { word-break: break-word; }
.slad-sum .empty { color: #898781; font-size: 0.85rem; }
.slad-ev td.w { white-space: nowrap; width: 150px; }
.slad-ev .bar { display: inline-block; width: 86px; height: 6px; background: #f0efec; border-radius: 3px;
  margin-right: 8px; vertical-align: middle; overflow: hidden; }
.slad-ev .bar > span { display: block; height: 6px; border-radius: 3px; }
"""


def evidence_html(evidence: Iterable[Evidence], limit: int | None = None, max_weight: float | None = None) -> str:
    """Tabela przesłanek jako HTML (zawijany tekst — nic się nie chowa w wąskich kolumnach).

    Kolumny: Grupa | Przesłanka | Wartość | Waga; grupa w kolorze (+ ikona i nazwa — nie tylko kolor),
    waga jako pasek. Wszystkie wartości z materiałów przechodzą przez ``html.escape``.
    """
    import html as _html

    rows = sorted(evidence, key=lambda e: -(e.weight or 0.0))
    if limit:
        rows = rows[:limit]
    if not rows:
        return ""
    top = max_weight or max((float(e.weight or 0.0) for e in rows), default=1.0) or 1.0
    body: list[str] = []
    for e in rows:
        g = (e.group or "").strip()
        color = GROUP_COLORS.get(g, INK_MUTED)
        w = float(e.weight or 0.0)
        pct = max(2.0, min(100.0, 100.0 * w / top)) if w > 0 else 0.0
        body.append(
            "<tr>"
            f'<td class="g" style="background:{tint(color, 0.22)}">{_html.escape(group_label(g))}</td>'
            f"<td>{_html.escape(clean_text(e.detail) or kind_label(e.kind))}</td>"
            f'<td class="{"v" if _is_identifier(evidence_value(e)) else "t"}">{_html.escape(evidence_value(e))}</td>'
            f'<td class="w"><span class="bar"><span style="width:{pct:.0f}%;background:{color}"></span></span>'
            f"{w:.1f}</td>"
            "</tr>"
        )
    head = "".join(f"<th>{c}</th>" for c in EVIDENCE_COLUMNS)
    return f'<table class="slad-ev"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def group_cell_style(label: Any) -> str:
    g = group_from_label(str(label))
    if g is None:
        return ""
    return f"background-color: {tint(GROUP_COLORS[g], 0.22)}; color: {INK}; font-weight: 600"


def style_groups(df: pd.DataFrame, column: str = "Grupa") -> Any:
    """Styler z kolorowaniem kolumny grupy (jasny odcień koloru grupy + ciemny tekst)."""
    if df.empty or column not in df.columns:
        return df
    try:
        return df.style.map(group_cell_style, subset=[column])
    except Exception:  # noqa: BLE001 — brak jinja2 itp.: tabela bez kolorów
        return df


def group_contributions(evidence: Iterable[Evidence]) -> list[tuple[str, float, float]]:
    """Wkład grup cech w wynik: [(grupa, suma wag, udział)] malejąco."""
    acc: dict[str, float] = defaultdict(float)
    for e in evidence:
        acc[e.group or "inne"] += max(0.0, float(e.weight or 0.0))
    total = sum(acc.values())
    out = [(g, w, (w / total if total else 0.0)) for g, w in acc.items()]
    return sorted(out, key=lambda t: -t[1])


def link_sentence(link: Link) -> str:
    """Jednozdaniowe uzasadnienie powiązania (po polsku)."""
    rel = relation_value(link.relation)
    s = f"Wynik {link.score:.1f} → relacja „{rel}”"
    if link.coverage:
        s += f", pokrycie cech {link.coverage:.0%}"
    parts = [f"{g} {w:.1f} ({share:.0%})" for g, w, share in group_contributions(link.evidence)[:3]]
    if parts:
        s += ". Najwięcej wnoszą: " + ", ".join(parts)
    return s + "."


def aggregate_evidence(links: Iterable[Link], top: int = 15) -> pd.DataFrame:
    """Przesłanki zsumowane po wielu powiązaniach (np. wewnątrz kampanii)."""
    acc: dict[tuple[str, str, str], dict[str, Any]] = {}
    for lk in links:
        seen: set[tuple[str, str, str]] = set()
        for e in lk.evidence:
            key = (e.group or "inne", e.kind, str(e.value) if e.value not in (None, "") else "")
            if key in seen:
                continue
            seen.add(key)
            slot = acc.setdefault(key, {"detail": clean_text(e.detail) or kind_label(e.kind), "n": 0, "w": 0.0})
            slot["n"] += 1
            slot["w"] += float(e.weight or 0.0)
    rows = [
        {
            "Grupa": group_label(g),
            "Przesłanka": v["detail"] if (val == "" or kind not in KIND_INFO) else kind_label(kind),
            "Wartość": shorten(val, 80) or "—",
            "Powiązania": v["n"],
            "Łączna waga": round(v["w"], 1),
        }
        for (g, kind, val), v in acc.items()
    ]
    rows.sort(key=lambda r: (-r["Łączna waga"], -r["Powiązania"]))
    return pd.DataFrame(rows[:top], columns=["Grupa", "Przesłanka", "Wartość", "Powiązania", "Łączna waga"])


def other_end(link: Link, mid: str) -> str:
    return link.b if link.a == mid else link.a


def links_frame(links: Iterable[Link], bundle: Bundle | None = None, top_evidence: int = 3) -> pd.DataFrame:
    rows = []
    for lk in sorted(links, key=lambda x: -x.score):
        top = sorted(lk.evidence, key=lambda e: -(e.weight or 0.0))[:top_evidence]
        rows.append(
            {
                "Materiał A": bundle.label(lk.a, 48) if bundle else lk.a,
                "Materiał B": bundle.label(lk.b, 48) if bundle else lk.b,
                "Relacja": relation_value(lk.relation),
                "Wynik": round(lk.score, 2),
                "Pokrycie": round(lk.coverage, 2),
                "Główne przesłanki": "; ".join(shorten(e.detail, 70) for e in top),
            }
        )
    cols = ["Materiał A", "Materiał B", "Relacja", "Wynik", "Pokrycie", "Główne przesłanki"]
    return pd.DataFrame(rows, columns=cols)


def nearest_frame(links: Iterable[Link], mid: str, bundle: Bundle | None = None) -> pd.DataFrame:
    rows = []
    for lk in sorted(links, key=lambda x: -x.score):
        other = other_end(lk, mid)
        top = max(lk.evidence, key=lambda e: e.weight or 0.0, default=None)
        rows.append(
            {
                "Materiał": bundle.label(other, 60) if bundle else other,
                "Kampania": (bundle.campaign_label(bundle.campaign_of.get(other)) if bundle else ""),
                "Relacja": relation_value(lk.relation),
                "Wynik": round(lk.score, 2),
                "Pokrycie": round(lk.coverage, 2),
                "Najsilniejsza przesłanka": shorten(top.detail, 90) if top else "—",
            }
        )
    cols = ["Materiał", "Kampania", "Relacja", "Wynik", "Pokrycie", "Najsilniejsza przesłanka"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------- przegląd


def overview_counts(b: Bundle) -> dict[str, int]:
    rel = Counter(relation_value(lk.relation) for lk in b.links)
    linked = {x for lk in b.links for x in (lk.a, lk.b)}
    return {
        "materials": len(b.materials),
        "campaigns": len(b.campaigns),
        "links": len(b.links),
        "duplikat": rel.get("duplikat", 0),
        "wariant": rel.get("wariant", 0),
        "powiązany": rel.get("powiązany", 0),
        "unlinked": sum(1 for mid in b.materials if mid not in linked),
        "unassigned": sum(1 for mid in b.materials if mid not in b.campaign_of),
        "with_features": sum(1 for mid in b.materials if mid in b.features),
        "labeled": sum(1 for m in b.materials.values() if m.label),
    }


def coverage_frame(b: Bundle) -> pd.DataFrame:
    n = len(b.materials) or 1
    counts = Counter(g for mid in b.materials if mid in b.features for g in set(b.features[mid].available))
    rows = [
        {"Grupa cech": label, "Materiały": counts.get(g, 0), "Udział": round(100.0 * counts.get(g, 0) / n, 1)}
        for g, label in COVERAGE_GROUPS
    ]
    return pd.DataFrame(rows, columns=["Grupa cech", "Materiały", "Udział"])


def _values_of(fs: FeatureSet, what: str) -> set[str]:
    t: TextFeatures = fs.text
    if what == "persons":
        vals = set(t.persons)
        vals |= {f.identity for img in fs.images for f in img.faces if f.identity}
        return vals
    if what == "brands":
        return set(t.brands)
    if what == "narratives":
        return set(t.narratives)
    if what == "domains":
        return set(fs.infra.domains) | set((t.iocs or {}).get("domain", []))
    if what == "trackers":
        return set(fs.infra.tracker_ids)
    return set()


TOP_LABELS: dict[str, str] = {
    "persons": "Osoba",
    "brands": "Marka / instytucja",
    "narratives": "Narracja",
    "domains": "Domena",
}


def top_values(features: Iterable[FeatureSet], what: str, n: int = 10) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for fs in features:
        counter.update(v for v in _values_of(fs, what) if v)
    label = TOP_LABELS.get(what, what)
    return pd.DataFrame(counter.most_common(n), columns=[label, "Materiały"])


# ---------------------------------------------------------------- kampanie


def _first(pairs: Sequence[tuple[str, int]] | None) -> str:
    return str(pairs[0][0]) if pairs else ""


def campaigns_frame(campaigns: Iterable[Campaign]) -> pd.DataFrame:
    rows = [
        {
            "ID": c.id,
            "Nazwa": clean_text(c.name),
            "Materiały": len(c.members),
            "Spójność": round(float(c.cohesion or 0.0), 2),
            "Główna osoba": _first(c.summary.persons),
            "Główna narracja": _first(c.summary.narratives),
            "Główna domena": _first(c.summary.domains),
        }
        for c in campaigns
    ]
    cols = ["ID", "Nazwa", "Materiały", "Spójność", "Główna osoba", "Główna narracja", "Główna domena"]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df = df.sort_values(["Materiały", "ID"], ascending=[False, True], kind="stable").reset_index(drop=True)
    return df


def _is_identifier(value: str) -> bool:
    return bool(value) and value != "—" and not re.search(r"\s", value)


def summary_html(blocks: Sequence[tuple[str, Sequence[tuple[str, int]] | None]], limit: int = 8) -> str:
    """Podsumowanie kampanii (osoby, marki, narracje, domeny, wskaźniki) jako siatka list „liczba · wartość”."""
    import html as _html

    boxes: list[str] = []
    for title, pairs in blocks:
        items = [
            f'<div class="it"><span class="n">{int(n)}</span><span class="x">{_html.escape(clean_text(v))}</span></div>'
            for v, n in list(pairs or [])[:limit]
        ]
        more = len(pairs or []) - limit
        if more > 0:
            items.append(f'<div class="empty">… oraz {more} kolejnych</div>')
        body = "".join(items) or '<div class="empty">brak danych</div>'
        boxes.append(f'<div class="box"><div class="h">{_html.escape(title)}</div>{body}</div>')
    return f'<div class="slad-sum">{"".join(boxes)}</div>'


def pairs_frame(
    pairs: Sequence[tuple[str, int]] | None, label: str, pretty: Callable[[str], str] | None = None
) -> pd.DataFrame:
    rows = [{label: (pretty(str(v)) if pretty else str(v)), "Materiały": int(n)} for v, n in (pairs or [])]
    return pd.DataFrame(rows, columns=[label, "Materiały"])


INDICATOR_PREFIXES: dict[str, str] = {
    "tracker_id": "tracker",
    "tracker": "tracker",
    "domain": "domena",
    "host": "host",
    "ip": "IP",
    "asn": "ASN",
    "favicon": "favicon",
    "tls_fingerprint": "certyfikat TLS",
    "nameserver": "NS",
    "form_action": "formularz",
    "form_fields": "pola formularza",
    "redirect_domain": "przekierowanie przez",
    "cert_domain": "domena w certyfikacie",
    "crypto_wallet": "portfel",
    "phone": "telefon",
    "email": "e-mail",
    "telegram": "Telegram",
    "whatsapp": "WhatsApp",
    "dom_simhash": "szablon DOM",
    "registrar_created": "rejestracja",
    "title": "tytuł",
}


def pretty_indicator(value: str) -> str:
    kind, sep, rest = str(value).partition(":")
    if sep and kind in INDICATOR_PREFIXES:
        return f"{INDICATOR_PREFIXES[kind]}: {rest}"
    return str(value)


def campaign_links(b: Bundle, campaign: Campaign) -> list[Link]:
    members = set(campaign.members)
    return [lk for lk in b.links if lk.a in members and lk.b in members]


# ---------------------------------------------------------------- materiał


def text_facts(fs: FeatureSet) -> pd.DataFrame:
    t = fs.text
    faces = sorted({f.identity for img in fs.images for f in img.faces if f.identity})
    rows = [
        ("Osoby", join_values(t.persons)),
        ("Twarze rozpoznane na obrazach", join_values(faces)),
        ("Marki / instytucje", join_values(t.brands)),
        ("Narracje", join_values(t.narratives)),
        ("Produkt", clean_text(t.product)),
        ("Obietnica", clean_text(t.promise)),
        ("Wezwanie do działania", clean_text(t.cta)),
        ("Słowa kluczowe", join_values(t.keywords, limit=20)),
        ("Język", clean_text(t.lang)),
        ("Wzbogacenie LLM", "tak" if t.llm_used else "nie (heurystyki)"),
    ]
    return pd.DataFrame([{"Cecha": k, "Wartość": v} for k, v in rows if v], columns=["Cecha", "Wartość"])


def iocs_frame(iocs: Mapping[str, Sequence[str]] | None) -> pd.DataFrame:
    rows = []
    for key, values in (iocs or {}).items():
        for v in values or []:
            rows.append({"Typ": IOC_LABELS.get(key, key), "Wartość": clean_text(v)})
    return pd.DataFrame(rows, columns=["Typ", "Wartość"])


def domain_age_days(inf: InfraFeatures, observed: datetime | None = None) -> int | None:
    if inf.domain_created is None:
        return None
    ref = inf.fetched_at or observed
    if ref is None:
        return None
    try:
        a, b = ref, inf.domain_created
        if (a.tzinfo is None) != (b.tzinfo is None):
            a, b = a.replace(tzinfo=None), b.replace(tzinfo=None)
        return (a - b).days
    except (TypeError, ValueError):
        return None


def infra_frame(inf: InfraFeatures, observed: datetime | None = None) -> pd.DataFrame:
    created = fmt_dt(inf.domain_created)
    age = domain_age_days(inf, observed)
    if created and age is not None:
        created += f" (domena miała {age} dni)"
    rows = [
        ("Adresy wejściowe", join_values(inf.input_urls, limit=5)),
        ("Adres końcowy", clean_text(inf.final_url)),
        ("Status HTTP", str(inf.http_status) if inf.http_status is not None else ""),
        ("Hosty", join_values(inf.hosts)),
        ("Domeny", join_values(inf.domains)),
        ("Tytuł strony", clean_text(inf.title)),
        ("Opis (meta)", shorten(inf.meta_description, 200)),
        ("Trackery", join_values(inf.tracker_ids)),
        ("Domeny skryptów", join_values(inf.script_domains)),
        ("Formularze (action)", join_values(inf.form_actions)),
        ("Pola formularzy", join_values(inf.form_fields, limit=20)),
        ("Linki wychodzące (domeny)", join_values(inf.outlinks)),
        ("Favicon (mmh3)", str(inf.favicon_mmh3) if inf.favicon_mmh3 is not None else ""),
        ("Szablon DOM (simhash)", clean_text(inf.dom_simhash)),
        ("SHA-256 HTML", clean_text(inf.html_sha256)),
        ("Adresy IP", join_values(inf.ips)),
        ("ASN", join_values(inf.asns)),
        ("Serwery nazw (NS)", join_values(inf.nameservers)),
        ("MX", join_values(inf.mx)),
        ("Rejestrator", clean_text(inf.registrar)),
        ("Data rejestracji domeny", created),
        ("Wystawca TLS", clean_text(inf.tls_issuer)),
        ("Odcisk certyfikatu TLS", clean_text(inf.tls_fingerprint)),
        ("Nazwy w certyfikacie (SAN)", join_values(inf.tls_sans)),
        ("Inne domeny z CT (crt.sh)", join_values(inf.ct_domains)),
        ("Pobrano", fmt_dt(inf.fetched_at, with_time=True)),
        ("Z pamięci podręcznej", "tak" if inf.from_cache else ""),
    ]
    return pd.DataFrame([{"Cecha": k, "Wartość": v} for k, v in rows if v], columns=["Cecha", "Wartość"])


def redirect_frame(inf: InfraFeatures) -> pd.DataFrame:
    chain = list(inf.redirect_chain) or ([inf.final_url] if inf.final_url else [])
    return pd.DataFrame(
        [{"Krok": i + 1, "Adres": clean_text(u)} for i, u in enumerate(chain)], columns=["Krok", "Adres"]
    )


def image_frame(images: Iterable[ImageFeatures]) -> pd.DataFrame:
    rows = [
        {
            "Plik": Path(img.path).name,
            "Rozmiar": f"{img.width}×{img.height}",
            "pHash": img.phash,
            "dHash": img.dhash,
            "wHash": img.whash,
            "colorHash": img.colorhash,
            "Twarze": len(img.faces),
            "SHA-256": (img.sha256 or "")[:16] + "…" if img.sha256 else "",
        }
        for img in images
    ]
    cols = ["Plik", "Rozmiar", "pHash", "dHash", "wHash", "colorHash", "Twarze", "SHA-256"]
    return pd.DataFrame(rows, columns=cols)


def faces_frame(images: Iterable[ImageFeatures]) -> pd.DataFrame:
    rows = []
    for img in images:
        for f in img.faces:
            rows.append(
                {
                    "Obraz": Path(img.path).name,
                    "Tożsamość": f.identity or "nieznana",
                    "Zgodność z bazą": round(f.identity_score, 3) if f.identity_score is not None else None,
                    "Pewność detekcji": round(f.score, 3),
                    "Ramka (x, y, w, h)": ", ".join(str(int(v)) for v in f.bbox),
                }
            )
    cols = ["Obraz", "Tożsamość", "Zgodność z bazą", "Pewność detekcji", "Ramka (x, y, w, h)"]
    return pd.DataFrame(rows, columns=cols)


def features_summary(fs: FeatureSet) -> pd.DataFrame:
    """Najważniejsze cechy w jednej tabeli (analiza nowego materiału)."""
    t, inf = fs.text, fs.infra
    faces = sorted({f.identity for img in fs.images for f in img.faces if f.identity})
    n_faces = sum(len(img.faces) for img in fs.images)
    ioc_counts = ", ".join(f"{IOC_LABELS.get(k, k)}: {len(v)}" for k, v in (t.iocs or {}).items() if v)
    registrar = clean_text(inf.registrar)
    if inf.domain_created:
        registrar = (registrar + " · " if registrar else "") + f"rejestracja {fmt_dt(inf.domain_created)}"
    rows = [
        ("Dostępne grupy cech", join_values(fs.available)),
        ("Osoby", join_values(t.persons)),
        ("Twarze", (f"{n_faces} (rozpoznane: {join_values(faces)})" if faces else str(n_faces)) if n_faces else ""),
        ("Marki / instytucje", join_values(t.brands)),
        ("Narracje", join_values(t.narratives)),
        ("Obietnica", clean_text(t.promise)),
        ("Wezwanie do działania", clean_text(t.cta)),
        ("Wskaźniki IOC", ioc_counts),
        ("Obrazy", str(len(fs.images)) if fs.images else ""),
        ("Adres końcowy", clean_text(inf.final_url)),
        ("Domeny", join_values(inf.domains)),
        ("Adresy IP / ASN", join_values([*inf.ips, *inf.asns])),
        ("Trackery", join_values(inf.tracker_ids)),
        ("Favicon (mmh3)", str(inf.favicon_mmh3) if inf.favicon_mmh3 is not None else ""),
        ("Rejestrator", registrar),
        ("Błędy / braki", str(len(fs.errors) + len(inf.errors)) if (fs.errors or inf.errors) else ""),
    ]
    return pd.DataFrame([{"Cecha": k, "Wartość": v} for k, v in rows if v], columns=["Cecha", "Wartość"])


def all_errors(fs: FeatureSet | None) -> list[str]:
    if fs is None:
        return []
    return [clean_text(e) for e in [*fs.errors, *fs.infra.errors] if clean_text(e)]


# ---------------------------------------------------------------- analiza nowego materiału


def status_view(assignment: Assignment) -> tuple[str, str, str]:
    key = relation_value(assignment.status)
    return STATUS_VIEW.get(key, (key, "gray", ":material/help:"))


def fallback_material(
    url: str | None = None,
    image_path: str | Path | None = None,
    html_path: str | Path | None = None,
    text: str | None = None,
) -> Material:
    """Materiał z wejścia użytkownika, gdy ``slad.ingest.material_from_input`` jest niedostępny."""
    h = hashlib.sha256()
    for part in (url or "", str(image_path or ""), str(html_path or ""), text or ""):
        h.update(part.encode("utf-8", "ignore") + b"\x00")
    for p in (image_path, html_path):
        if p:
            try:
                h.update(Path(p).read_bytes())
            except OSError:
                pass
    if image_path:
        source = SourceKind.IMAGE
    elif html_path:
        source = SourceKind.HTML
    elif url:
        source = SourceKind.URL
    else:
        source = SourceKind.TEXT
    return Material(
        id="nowy-" + h.hexdigest()[:12],
        source=source,
        url=(url or None),
        text=(text or None),
        image_paths=[str(Path(image_path).resolve())] if image_path else [],
        html_path=str(Path(html_path).resolve()) if html_path else None,
    )


# ---------------------------------------------------------------- postęp przetwarzania


def parse_progress(*args: Any, **kwargs: Any) -> tuple[float, str]:
    """Normalizuje dowolny styl wywołania callbacku postępu → (ułamek 0..1, opis).

    Obsługuje m.in. ``(done, total)``, ``(done, total, "etap")``, ``("etap", done, total)``,
    ``(0.4,)``, ``(40,)`` (procent) i argumenty nazwane ``done/total/stage/message/fraction``.
    """
    nums = [a for a in args if isinstance(a, int | float) and not isinstance(a, bool)]
    texts = [a for a in args if isinstance(a, str)]
    done = next((kwargs[k] for k in ("done", "current", "i", "step", "n_done") if k in kwargs), None)
    total = next((kwargs[k] for k in ("total", "n", "count", "n_total") if k in kwargs), None)
    frac = next((kwargs[k] for k in ("fraction", "frac", "progress", "value") if k in kwargs), None)
    text = next((str(kwargs[k]) for k in ("stage", "message", "msg", "text", "desc", "label") if kwargs.get(k)), "")
    if not text and texts:
        text = texts[0]
    if frac is None and done is None and total is None:
        if len(nums) >= 2:
            done, total = nums[0], nums[1]
        elif len(nums) == 1:
            v = float(nums[0])
            frac = v if 0.0 <= v <= 1.0 else v / 100.0
    if frac is None and done is not None and total:
        try:
            frac = float(done) / float(total)
        except (TypeError, ValueError, ZeroDivisionError):
            frac = None
    try:
        frac_f = float(frac) if frac is not None else 0.0
    except (TypeError, ValueError):
        frac_f = 0.0
    if math.isnan(frac_f):
        frac_f = 0.0
    frac_f = min(1.0, max(0.0, frac_f))
    if done is not None and total:
        counter = f"{done}/{total}"
        text = f"{text} ({counter})" if text else counter
    return frac_f, text


# ---------------------------------------------------------------- metodyka i ustawienia


def weights_frame(weights: Mapping[str, float]) -> pd.DataFrame:
    rows = [
        {
            "Grupa": group_label(kind_group(k)) if kind_group(k) != "—" else "—",
            "Przesłanka": kind_label(k),
            "Klucz": k,
            "Waga": float(w),
        }
        for k, w in weights.items()
    ]
    df = pd.DataFrame(rows, columns=["Grupa", "Przesłanka", "Klucz", "Waga"])
    if not df.empty:
        df = df.sort_values(["Waga", "Klucz"], ascending=[False, True], kind="stable").reset_index(drop=True)
    return df


def thresholds_frame(thresholds: Mapping[str, float]) -> pd.DataFrame:
    names = {"duplicate": "duplikat", "variant": "wariant", "related": "powiązany"}
    rows = [{"Relacja": names.get(k, k), "Minimalny wynik": float(v)} for k, v in thresholds.items()]
    df = pd.DataFrame(rows, columns=["Relacja", "Minimalny wynik"])
    if not df.empty:
        df = df.sort_values("Minimalny wynik", ascending=False, kind="stable").reset_index(drop=True)
    return df


def settings_frame(cfg: Any) -> pd.DataFrame:
    """Najważniejsze ustawienia (bez sekretów — klucz API nigdy nie trafia do UI)."""

    def g(path: str, default: Any = None) -> Any:
        obj = cfg
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return default
        return obj

    def yn(v: Any) -> str:
        return "tak" if v else "nie"

    rows = [
        ("Ważenie rzadkością (IDF)", yn(g("link.rarity"))),
        ("Metoda grupowania", str(g("link.cluster.method", ""))),
        ("Rozdzielczość Louvain", str(g("link.cluster.resolution", ""))),
        ("Minimalny rozmiar kampanii", str(g("link.cluster.min_size", ""))),
        ("Przypisanie: minimalny wynik", str(g("link.assign.min_score", ""))),
        ("Przypisanie: liczba sąsiadów (top-k)", str(g("link.assign.top_k", ""))),
        ("Sieć (pobieranie stron, DNS, RDAP, TLS)", yn(g("network.enabled"))),
        ("Certificate Transparency (crt.sh)", yn(g("network.crtsh"))),
        ("OCR", yn(g("ocr.enabled"))),
        ("LLM", (yn(g("llm.enabled")) + (f" · model {g('llm.model')}" if g("llm.model") else ""))),
        ("Rozpoznawanie twarzy", yn(g("faces.enabled"))),
        ("CLIP", yn(g("clip.enabled"))),
        ("MinHash: permutacje / shingle", f"{g('text.minhash_perm', '')} / {g('text.shingle', '')}"),
    ]
    return pd.DataFrame([{"Ustawienie": k, "Wartość": v} for k, v in rows], columns=["Ustawienie", "Wartość"])


BUILTIN_METHODOLOGY = """
### Jak Bloodhound łączy materiały w kampanie

1. **Pozyskanie cech.** Każdy materiał (URL, zrzut ekranu, zapisany HTML, tekst, rekord zbioru) jest
   opisywany trzema grupami cech:
   - **tekst** — OCR zrzutów, tekst strony docelowej i wejścia; osoby, marki, narracja, obietnica zysku,
     wezwanie do działania, wskaźniki IOC (telefony, e-maile, portfele krypto, Telegram/WhatsApp, IBAN),
     sygnatura MinHash;
   - **obraz** — hashe percepcyjne (pHash/dHash/wHash/colorHash), twarze rozpoznawane względem bazy
     referencyjnej;
   - **infrastruktura** — łańcuch przekierowań, domeny, trackery (GA/GTM/Pixel), favicona, szablon DOM,
     DNS, RDAP, IP/ASN, certyfikat TLS.
2. **Podobieństwo par.** Dla kandydatów (wspólne wskaźniki, LSH MinHash, kubełki pHash) liczymy
   **przesłanki**. Wynik powiązania to suma wag przesłanek; wskaźniki częste w zbiorze (np. wspólny CDN)
   są osłabiane (ważenie rzadkością). Porównujemy tylko cechy dostępne w obu materiałach.
3. **Relacja.** Progi wyniku wyznaczają relację: *duplikat*, *wariant* albo *powiązany*.
4. **Grupowanie.** Kampanie to spójne składowe grafu silnych powiązań, dzielone algorytmem Louvain,
   aby uniknąć „łańcuchowania” niepowiązanych materiałów.
5. **Nowy materiał** jest porównywany z bazą: przypisanie do kampanii, sygnał „potencjalnie nowa
   kampania” albo brak powiązań — zawsze z listą przesłanek.
6. **Niekompletne dane** (martwa strona, brak metadanych) obniżają pokrycie, ale nie blokują analizy.
"""


# ---------------------------------------------------------------- raport ewaluacji i eksport


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, str | int | float | bool)


def report_tables(report: Mapping[str, Any]) -> tuple[pd.DataFrame, list[tuple[str, pd.DataFrame]]]:
    """Dowolny słownik raportu → (tabela metryk skalarnych, [(tytuł, tabela)] dla struktur)."""
    scalars: list[dict[str, Any]] = []
    tables: list[tuple[str, pd.DataFrame]] = []

    def fmt(v: Any) -> Any:
        return round(v, 4) if isinstance(v, float) else v

    def walk(prefix: str, obj: Any) -> None:
        if _is_scalar(obj):
            scalars.append({"Metryka": prefix, "Wartość": str(fmt(obj))})
        elif isinstance(obj, Mapping):
            if obj and all(_is_scalar(v) for v in obj.values()):
                for k, v in obj.items():
                    scalars.append({"Metryka": f"{prefix}.{k}" if prefix else str(k), "Wartość": str(fmt(v))})
            elif obj and all(isinstance(v, Mapping) and all(_is_scalar(x) for x in v.values()) for v in obj.values()):
                df = pd.DataFrame.from_dict({str(k): dict(v) for k, v in obj.items()}, orient="index")
                tables.append((prefix or "raport", df.map(fmt) if hasattr(df, "map") else df))
            else:
                for k, v in obj.items():
                    walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(obj, list | tuple):
            if obj and all(isinstance(x, Mapping) for x in obj):
                tables.append((prefix or "raport", pd.DataFrame([dict(x) for x in obj])))
            elif all(_is_scalar(x) for x in obj):
                scalars.append({"Metryka": prefix, "Wartość": join_values(obj, limit=30)})
            else:
                tables.append((prefix or "raport", pd.DataFrame({"wartość": [str(x) for x in obj]})))
        else:
            scalars.append({"Metryka": prefix, "Wartość": str(obj)})

    walk("", dict(report))
    return pd.DataFrame(scalars, columns=["Metryka", "Wartość"]), tables


def zip_dir(directory: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(directory.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(directory).as_posix())
    return buf.getvalue()


def zip_frames(frames: Mapping[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, df in frames.items():
            zf.writestr(name, df.to_csv(index=False))
    return buf.getvalue()


def assignments_frame(b: Bundle) -> pd.DataFrame:
    rows = []
    for mid, m in b.materials.items():
        cid = b.campaign_of.get(mid)
        rows.append(
            {
                "material_id": mid,
                "campaign_id": cid or "",
                "campaign_name": b.campaign_by_id[cid].name if cid in b.campaign_by_id else "",
                "url": m.url or "",
                "source": relation_value(m.source),
                "links": len(b.links_by_material.get(mid, [])),
            }
        )
    return pd.DataFrame(rows, columns=["material_id", "campaign_id", "campaign_name", "url", "source", "links"])
