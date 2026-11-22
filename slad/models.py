"""Model danych Bloodhound — wspólny kontrakt wszystkich modułów.

Przepływ:  Material ─► FeatureSet ─► Link (z przesłankami Evidence) ─► Campaign
                                      └─► Assignment (analiza nowego materiału)

Zasady:
- Material.label służy WYŁĄCZNIE do ewaluacji — żaden moduł cech/powiązań nie może go czytać.
- Brak danych to nie błąd: puste listy / None + wpis w `errors`; porównania liczone tylko na
  dostępnych cechach (odporność na niekompletne dane jest kryterium oceny NASK).
- Teksty widoczne w UI (Evidence.detail, Campaign.name, Assignment.explanation) — po polsku.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- wejście


class SourceKind(str, Enum):
    URL = "url"  # adres reklamy / strony docelowej
    IMAGE = "image"  # zrzut ekranu / kreacja graficzna
    HTML = "html"  # zapisany kod strony
    TEXT = "text"  # sam tekst (treść reklamy, log komunikacji)
    RECORD = "record"  # rekord ze zbioru (wiele pól naraz)


class Material(BaseModel):
    """Jeden materiał wejściowy: reklama, strona, zrzut ekranu albo rekord zbioru."""

    id: str  # stabilny identyfikator (z rekordu albo skrót treści)
    source: SourceKind
    url: str | None = None  # adres reklamy lub strony docelowej podany na wejściu
    text: str | None = None  # tekst podany na wejściu (treść reklamy, opis, log)
    image_paths: list[str] = Field(default_factory=list)  # ścieżki absolutne
    html_path: str | None = None  # zapisany HTML strony (ścieżka absolutna)
    platform: str | None = None  # np. facebook / instagram / google / tiktok
    advertiser: str | None = None  # nazwa profilu / reklamodawcy, jeśli znana
    observed_at: datetime | None = None
    label: str | None = None  # etykieta kampanii — TYLKO ewaluacja, nigdy cecha
    # surowe pola rekordu; meta["infra"] = gotowe dane infrastruktury
    meta: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- cechy


class TextFeatures(BaseModel):
    input_text: str = ""  # tekst z wejścia (Material.text)
    ocr_text: str = ""  # tekst odczytany z obrazów (OCR)
    page_text: str = ""  # widoczny tekst strony docelowej
    text: str = ""  # złączony, znormalizowany tekst do porównań
    lang: str | None = None
    keywords: list[str] = Field(default_factory=list)
    persons: list[str] = Field(default_factory=list)  # kanoniczne nazwy osób (gazetteer / LLM)
    brands: list[str] = Field(default_factory=list)  # podszywane marki, media, instytucje
    narratives: list[str] = Field(default_factory=list)  # tagi narracji z config/narratives.yaml
    product: str | None = None  # np. nazwa „platformy inwestycyjnej”
    promise: str | None = None  # obietnica, np. „5000 zł dziennie”
    cta: str | None = None  # wezwanie do działania
    # klucze: phone, email, url, domain, crypto_btc, crypto_eth, crypto_trx, telegram, whatsapp, iban
    iocs: dict[str, list[str]] = Field(default_factory=dict)
    minhash: list[int] = Field(default_factory=list)  # sygnatura MinHash (pusta gdy brak tekstu)
    llm_used: bool = False


class Face(BaseModel):
    bbox: tuple[int, int, int, int]  # x, y, w, h
    score: float
    embedding: list[float] = Field(default_factory=list)  # L2-znormalizowany wektor twarzy
    identity: str | None = None  # osoba z bazy referencyjnej (reference/faces/<Osoba>/)
    identity_score: float | None = None


class ImageFeatures(BaseModel):
    path: str
    sha256: str
    width: int
    height: int
    phash: str  # hex, imagehash
    dhash: str
    whash: str
    colorhash: str
    dominant_colors: list[str] = Field(default_factory=list)  # "#rrggbb"
    faces: list[Face] = Field(default_factory=list)
    clip: list[float] = Field(default_factory=list)  # opcjonalny embedding semantyczny obrazu


class InfraFeatures(BaseModel):
    input_urls: list[str] = Field(default_factory=list)
    redirect_chain: list[str] = Field(default_factory=list)  # kolejne adresy (z wejściowym i końcowym)
    final_url: str | None = None
    http_status: int | None = None
    hosts: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)  # domeny zarejestrowane (eTLD+1) z łańcucha
    title: str | None = None
    meta_description: str | None = None
    html_sha256: str | None = None
    dom_simhash: str | None = None  # 64-bit hex — struktura DOM (szablon strony)
    # "ga:UA-1234-1", "gtag:G-XXXX", "gtm:GTM-XXXX", "fbpixel:123…", "ym:123", "tiktok:XXXX", "keitaro:…"
    tracker_ids: list[str] = Field(default_factory=list)
    script_domains: list[str] = Field(default_factory=list)
    form_actions: list[str] = Field(default_factory=list)
    form_fields: list[str] = Field(default_factory=list)
    outlinks: list[str] = Field(default_factory=list)  # domeny linków wychodzących
    favicon_mmh3: int | None = None  # hash favicony (styl Shodana)
    ips: list[str] = Field(default_factory=list)
    asns: list[str] = Field(default_factory=list)  # "AS13335 CLOUDFLARENET"
    nameservers: list[str] = Field(default_factory=list)
    mx: list[str] = Field(default_factory=list)
    registrar: str | None = None
    domain_created: datetime | None = None
    tls_issuer: str | None = None
    tls_fingerprint: str | None = None  # sha256 certyfikatu
    tls_sans: list[str] = Field(default_factory=list)
    ct_domains: list[str] = Field(default_factory=list)  # inne domeny z Certificate Transparency
    errors: list[str] = Field(default_factory=list)  # co się nie udało — niekompletne dane
    fetched_at: datetime | None = None
    from_cache: bool = False


class FeatureSet(BaseModel):
    material_id: str
    text: TextFeatures = Field(default_factory=TextFeatures)
    images: list[ImageFeatures] = Field(default_factory=list)
    infra: InfraFeatures = Field(default_factory=InfraFeatures)
    available: list[str] = Field(default_factory=list)  # "tekst" | "obraz" | "twarze" | "infrastruktura"
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- powiązania i kampanie

GROUPS = ("tekst", "obraz", "osoby", "infrastruktura", "narracja")


class Evidence(BaseModel):
    """Jedna przesłanka powiązania dwóch materiałów — podstawa interpretowalności."""

    kind: str  # klucz wagi z config/settings.yaml → link.weights (np. "tracker_id", "image_phash")
    group: str  # jedna z GROUPS
    detail: str  # opis po polsku, np. „wspólny identyfikator Facebook Pixel 1234567890”
    value: str | None = None  # wspólna wartość (jeśli dotyczy)
    similarity: float | None = None  # 0..1 dla cech ciągłych
    weight: float  # wkład w wynik po ważeniu rzadkością


class Relation(str, Enum):
    DUPLICATE = "duplikat"
    VARIANT = "wariant"
    RELATED = "powiązany"


class Link(BaseModel):
    a: str
    b: str
    score: float  # suma wag przesłanek
    relation: Relation
    evidence: list[Evidence] = Field(default_factory=list)
    coverage: float = 0.0  # 0..1 — część grup cech dostępnych w OBU materiałach


class CampaignSummary(BaseModel):
    persons: list[tuple[str, int]] = Field(default_factory=list)  # (wartość, liczba materiałów)
    brands: list[tuple[str, int]] = Field(default_factory=list)
    narratives: list[tuple[str, int]] = Field(default_factory=list)
    domains: list[tuple[str, int]] = Field(default_factory=list)
    indicators: list[tuple[str, int]] = Field(default_factory=list)  # wspólne wskaźniki infrastruktury
    keywords: list[str] = Field(default_factory=list)
    first_seen: datetime | None = None
    last_seen: datetime | None = None


class Campaign(BaseModel):
    id: str  # "K001", "K002", …
    name: str  # automatyczna nazwa po polsku
    members: list[str]
    cohesion: float = 0.0  # średni wynik powiązań wewnątrz kampanii
    summary: CampaignSummary = Field(default_factory=CampaignSummary)
    key_evidence: list[Evidence] = Field(default_factory=list)  # najczęstsze silne przesłanki


class AssignmentStatus(str, Enum):
    ASSIGNED = "przypisany"  # pasuje do istniejącej kampanii
    NEW_CAMPAIGN = "nowa_kampania"  # powiązany z materiałami spoza kampanii → kandydat na nową
    UNLINKED = "brak_powiązań"  # nic podobnego w bazie


class Assignment(BaseModel):
    material_id: str
    campaign_id: str | None = None
    score: float = 0.0
    status: AssignmentStatus = AssignmentStatus.UNLINKED
    nearest: list[Link] = Field(default_factory=list)  # najbliższe materiały z przesłankami
    explanation: str = ""  # uzasadnienie po polsku


class CorpusStats(BaseModel):
    """Częstości wartości wskaźników w zbiorze — do ważenia rzadkością (wspólny CDN = słaba przesłanka)."""

    n: int = 0
    df: dict[str, dict[str, int]] = Field(default_factory=dict)  # kind → wartość → liczba materiałów
