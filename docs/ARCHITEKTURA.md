# Architektura Śladu Fejka

> Dokument jest jednocześnie **kontraktem między modułami**: sygnatury poniżej są wiążące.
> Wspólny model danych: [`slad/models.py`](../slad/models.py), konfiguracja: [`slad/settings.py`](../slad/settings.py) + [`config/settings.yaml`](../config/settings.yaml).
> Metoda (cechy, podobieństwo, grupowanie, wagi): [METODYKA.md](METODYKA.md).

## Przepływ

```
 wejście: zbiór NASK (CSV/JSON/XLSX + obrazy/HTML) · URL · zrzut ekranu · tekst
   │
   ▼  ingest/            Material  (jednolity model; etykieta kampanii tylko do ewaluacji)
   │
   ▼  features/          FeatureSet
   │    ├─ tekst         OCR obrazów (RapidOCR) + tekst strony + LLM/heurystyki:
   │    │                osoby, marki, narracja, produkt, obietnica, CTA, IOC (telefony, e-maile,
   │    │                portfele krypto, Telegram/WhatsApp, IBAN), słowa kluczowe, MinHash
   │    ├─ obraz         pHash/dHash/wHash/colorHash, kolory, twarze (YuNet + SFace) → tożsamość
   │    │                z bazy referencyjnej, opcjonalnie CLIP
   │    └─ infrastruktura łańcuch przekierowań, HTML (tytuł, trackery GA/GTM/Pixel, formularze,
   │                     szablon DOM – simhash), favicon (mmh3), DNS, RDAP/WHOIS, IP/ASN, TLS, CT
   │
   ▼  link/              Link + Evidence  (każde powiązanie ma przesłanki i wagi)
   │    kandydaci (indeks odwrotny wskaźników + LSH MinHash + kubełki pHash) → porównanie par →
   │    wynik = Σ wag przesłanek ważonych rzadkością → relacja: duplikat / wariant / powiązany
   │
   ▼  link/cluster       Campaign  (spójne składowe silnych powiązań + Louvain przeciw „łańcuchowaniu”)
   │
   ├─► link/assign       Assignment: nowy materiał → kampania / nowa kampania / brak powiązań
   ├─► ui/ (Streamlit)   przegląd, kampanie, graf, materiał, analiza nowego materiału, metodyka
   ├─► export/           Neo4j (CSV + Cypher), GraphML, JSON
   └─► eval/             metryki grupowania i przypisań (gdy znane etykiety)
```

## Moduły i sygnatury

| Plik | Publiczne API |
|---|---|
| `slad/models.py` | model danych (gotowy, zmiany tylko przez integrację) |
| `slad/settings.py` | `load_settings(path=None, **overrides) -> Settings`, `ROOT` |
| `slad/ingest/__init__.py` | `load_dataset(path: Path) -> list[Material]` · `material_from_input(url=None, image_path=None, html_path=None, text=None) -> Material` |
| `slad/ingest/loaders.py` | rozpoznawanie formatu: CSV / JSON / JSONL / XLSX / katalog obrazów / HTML / TXT / lista URL; mapowanie kolumn po synonimach (PL/EN) |
| `slad/ingest/synthetic.py` | `generate(out_dir: Path, seed: int = 7, n_campaigns: int = 8, noise: int = 12) -> Path` — zbiór syntetyczny z etykietami (testy, demo bez danych NASK) |
| `slad/features/__init__.py` | **gotowe:** `ExtractContext.from_settings(cfg, workspace=None)` (pola: `settings`, `cache_root`, `llm`, `faces`, `clip`, metoda `cache(name) -> JsonCache`) · `extract(material, ctx) -> FeatureSet` |
| `slad/features/ocr.py` | `ocr_image(path: Path, ctx: ExtractContext) -> str` (cache `ctx.cache("ocr")`, klucz = sha256 pliku) |
| `slad/features/iocs.py` | `extract_iocs(text: str) -> dict[str, list[str]]` |
| `slad/features/text.py` | `text_features(material, ocr_texts, page_text, ctx) -> TextFeatures` · `minhash_signature(text, cfg) -> list[int]` · `normalize_text(text) -> str` |
| `slad/features/llm.py` | `LLMClient.from_settings(cfg: LLMSettings, cache_dir) -> LLMClient \| None` · `enrich(text, client) -> dict \| None` |
| `slad/features/visual.py` | `image_features(path: Path, ctx) -> ImageFeatures` (nieczytelny obraz → `ValueError`) · `hamming(hex_a, hex_b) -> int` · `phash_similarity(a, b) -> float` |
| `slad/features/faces.py` | `FaceEngine.from_settings(cfg: FaceSettings) -> FaceEngine \| None` · `.detect(image) -> list[Face]` |
| `slad/features/clip.py` | `ClipEmbedder.from_settings(cfg: ClipSettings) -> ClipEmbedder \| None` · `.embed(image) -> list[float]` (opcjonalne) |
| `slad/features/infra/__init__.py` | `infra_features(material, ctx) -> tuple[InfraFeatures, str]` — drugi element to widoczny tekst strony docelowej (`page_text`) |
| `slad/features/infra/*.py` | `cache.JsonCache`, `fetch.fetch_chain`, `html.parse_html`, `netinfo.dns_info / rdap_domain / rdap_ip`, `tls.tls_info / crtsh_domains`, `favicon.favicon_hash`, `domains.registered_domain / normalize_url` |
| `slad/link/__init__.py` | `build(features: list[FeatureSet], cfg: Settings) -> tuple[list[Link], list[Campaign], CorpusStats]` |
| `slad/link/indicators.py` | `indicators(fs: FeatureSet) -> dict[str, set[str]]` |
| `slad/link/stats.py` | `corpus_stats(features) -> CorpusStats` · `rarity_factor(kind, value, stats) -> float` |
| `slad/link/evidence.py` | `LinkContext` · `compare(fa, fb, ctx) -> Link \| None` |
| `slad/link/candidates.py` | `candidate_pairs(features, ctx) -> set[tuple[str, str]]` |
| `slad/link/cluster.py` | `cluster(ids, links, cfg: LinkSettings) -> list[list[str]]` |
| `slad/link/summarize.py` | `summarize(cid, members, features_by_id, links) -> Campaign` |
| `slad/link/assign.py` | `assign(fs: FeatureSet, ws: Workspace, cfg: Settings) -> Assignment` |
| `slad/store.py` | **gotowe:** `Workspace(root)` — `materials() / material(id) / save_materials() / add_material()`, `features(id) / save_features() / all_features()`, `links() / save_links()`, `campaigns() / save_campaigns()`, `stats() / save_stats()`, `campaign_of(id)`, `cache_dir(name)`, `path(name)` |
| `slad/pipeline.py` | `process_dataset(path, ws, cfg, progress=None) -> None` · `analyze_new(material, ws, cfg) -> tuple[FeatureSet, Assignment]` |
| `slad/cli.py` | `python -m slad run / ingest / link / analyze / export / eval / synthetic / models / ui` |
| `slad/export/` | `neo4j.export_neo4j(ws, out_dir)` · `graph.export_graphml(ws, path)` · `graph.export_json(ws, path)` |
| `slad/eval/` | `metrics.pairwise_prf / bcubed / ari_nmi` · `run.evaluate(ws, cfg) -> dict` (+ leave-one-out przypisań) |
| `slad/ui/app.py` | `streamlit run slad/ui/app.py` |

## Katalog roboczy (workspace)

```
workspace/<nazwa>/
  materials.jsonl        Material — po jednym w wierszu
  features/<id>.json     FeatureSet
  links.jsonl            Link (z przesłankami)
  campaigns.json         list[Campaign]
  stats.json             CorpusStats
  text_index.pkl         indeks TF-IDF (podobieństwo treści, analiza nowych materiałów)
  cache/{infra,ocr,llm}/ pamięć podręczna (powtórne uruchomienie bez sieci)
```

## Zasady wspólne

- Python 3.13, CPU, bez GPU. Testy bez sieci (`pytest -q`), sieć tylko w module `features/infra` i tylko gdy `network.enabled`.
- Pobieranie stron oszustów: wyłącznie HTML (bez JavaScriptu), limit rozmiaru i czasu, bez zapisywania plików wykonywalnych.
- Każda brakująca cecha → puste pole + wpis w `errors`; porównanie liczone na cechach dostępnych w obu materiałach.
- Materiałów NASK nie publikujemy: `data/` i `workspace/` są w `.gitignore`.
