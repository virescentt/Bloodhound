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
| `slad/ingest/__init__.py` | `load_dataset(path, *, label_from_dir=False) -> list[Material]` (nieczytelny rekord → pominięty z ostrzeżeniem w `meta["ingest_warnings"]`) · `material_from_input(url=None, image_path=None, html_path=None, text=None) -> Material` (nigdy nie rzuca; id `new-…`) · `normalize_url`, `parse_date`, `map_column` |
| `slad/ingest/loaders.py` | rozpoznawanie formatu: CSV / JSON / JSONL / XLSX / katalog obrazów / HTML / TXT / lista URL; mapowanie kolumn po synonimach (PL/EN) |
| `slad/ingest/synthetic.py` | `generate(out_dir: Path, seed: int = 7, n_campaigns: int = 8, noise: int = 12) -> Path` — zbiór syntetyczny z etykietami (testy, demo bez danych NASK) |
| `slad/features/__init__.py` | **gotowe:** `ExtractContext.from_settings(cfg, workspace=None)` (pola: `settings`, `cache_root`, `llm`, `faces`, `clip`, metoda `cache(name) -> JsonCache`) · `extract(material, ctx) -> FeatureSet` |
| `slad/features/ocr.py` | `ocr_image(path: Path, ctx: ExtractContext) -> str` (cache `ctx.cache("ocr")`, klucz = sha256 pliku) |
| `slad/features/iocs.py` | `extract_iocs(text: str) -> dict[str, list[str]]` |
| `slad/features/text.py` | `text_features(material, ocr_texts, page_text, ctx) -> TextFeatures` · `minhash_signature(text, cfg) -> list[int]` · `normalize_text(text) -> str` · `fold(text)` · `find_person_names(text) -> list[str]` (heurystyka imię + nazwisko, `config/gazetteer/first_names.yaml`) |
| `slad/features/llm.py` | `LLMClient.from_settings(cfg: LLMSettings, cache_dir, *, probe=True) -> LLMClient \| None` · `enrich(text, client) -> dict \| None` (nigdy nie rzuca; json_schema → json_object → JSON w treści) |
| `slad/features/visual.py` | `image_features(path: Path, ctx) -> ImageFeatures` (nieczytelny obraz → `ValueError`) · `hamming(hex_a, hex_b) -> int` · `phash_similarity(a, b) -> float` |
| `slad/features/faces.py` | `FaceEngine.from_settings(cfg: FaceSettings) -> FaceEngine \| None` · `.detect(image) -> list[Face]` · `model_download.download_face_models(dest, force=False)` (`python -m slad models`) |
| `slad/features/clip.py` | `ClipEmbedder.from_settings(cfg: ClipSettings) -> ClipEmbedder \| None` · `.embed(image) -> list[float]` (opcjonalne) |
| `slad/features/infra/__init__.py` | `infra_features(material, ctx) -> tuple[InfraFeatures, str]` — drugi element to widoczny tekst strony docelowej (`page_text`) |
| `slad/features/infra/*.py` | `cache.JsonCache`, `fetch.fetch_chain`, `html.parse_html`, `netinfo.dns_info / rdap_domain / rdap_ip`, `tls.tls_info / crtsh_domains`, `favicon.favicon_hash`, `domains.registered_domain / normalize_url` |
| `slad/link/__init__.py` | `build(features, cfg, dates=None, progress=None) -> tuple[list[Link], list[Campaign], CorpusStats]` (kampanie K001… od największej) · `link_context(features, cfg)` |
| `slad/link/indicators.py` | `indicators(fs, stop=None) -> dict[str, set[str]]` · `DEFAULT_WEIGHTS`, `STRONG_KINDS`, `SOFT_KINDS` · `load_stoplists()` · `page_is_generic(fs)` |
| `slad/link/stats.py` | `corpus_stats(features) -> CorpusStats` · `rarity_factor(kind, value, stats) -> float` |
| `slad/link/evidence.py` | `LinkContext(cfg, features, stats=None)` (`.add(fs)`, `.text_sims`, `.segment_sims`) · `compare(fa, fb, ctx) -> Link \| None` · `score_pair(fa, fb, ctx) -> PairScore` (pełna ocena także poniżej progu — „dlaczego nie powiązano”) |
| `slad/link/candidates.py` | `candidate_pairs(features, ctx) -> set[tuple[str, str]]` |
| `slad/link/cluster.py` | `cluster(ids, links, cfg: LinkSettings) -> list[list[str]]` |
| `slad/link/summarize.py` | `summarize(cid, members, features_by_id, links, dates=None, indicators_by_id=None) -> Campaign` |
| `slad/link/assign.py` | `assign(fs: FeatureSet, ws: Workspace, cfg: Settings) -> Assignment` · `workspace_context(ws, cfg)` (kontekst bazy w pamięci, unieważniany zmianą plików workspace) |
| `slad/store.py` | **gotowe:** `Workspace(root)` — `materials() / material(id) / save_materials() / add_material()`, `features(id) / save_features() / all_features()`, `links() / save_links()`, `campaigns() / save_campaigns()`, `stats() / save_stats()`, `campaign_of(id)`, `cache_dir(name)`, `path(name)` |
| `slad/pipeline.py` | `process_dataset(path, ws, cfg, progress=None, limit=None, *, force=False, ctx=None, stage_progress=None) -> dict` (podsumowanie; przyrostowo) · `rebuild_links(ws, cfg)` · `analyze_new(material, ws, cfg, add=False, *, ctx=None) -> tuple[FeatureSet, Assignment]` |
| `slad/cli.py` | `python -m slad run / ingest / link / analyze / campaigns / export / eval / synthetic / models / ui / demo` (`--offline`, `--no-llm`, `--no-faces`, `-w`, `-j`) |
| `slad/export/` | `export_neo4j(ws, out_dir)` (CSV + `import.cypher`) · `export_graphml(ws, path)` · `export_json(ws, path)` |
| `slad/eval/` | `metrics.pairwise_prf / bcubed / ari_nmi` · `run.evaluate(ws, cfg=None) -> dict` · `run.format_report(report) -> str` (trafność wg relacji i rodzaju przesłanki, przypisania, „nie łącz niepowiązanych”) |
| `slad/ui/app.py` | `streamlit run slad/ui/app.py` (albo `python -m slad ui`); workspace z `SLAD_WORKSPACE` lub wybór w panelu |

## Katalog roboczy (workspace)

```
workspace/<nazwa>/
  materials.jsonl        Material — po jednym w wierszu
  features/<id>.json     FeatureSet
  links.jsonl            Link (z przesłankami)
  campaigns.json         list[Campaign]
  stats.json             CorpusStats
  run.json               podsumowanie ostatniego przetwarzania (UI: przegląd)
  report.json            raport ewaluacji (python -m slad eval)
  uploads/               pliki dodane w panelu (analiza nowego materiału)
  cache/<przestrzeń>/    pamięć podręczna: ocr, llm, fetch, dns, asn, rdap, tls, favicon, crtsh
```

Indeks TF-IDF nie jest zapisywany na dysku (`text_index.pkl` z pierwotnego kontraktu): `assign` buduje kontekst porównań z plików workspace i trzyma go w pamięci do czasu ich zmiany (3 000 materiałów: ~1.3 s raz, potem ~20 ms).

## Zasady wspólne

- Python 3.13, CPU, bez GPU. Testy bez sieci (`pytest -q`), sieć tylko w module `features/infra` i tylko gdy `network.enabled`.
- Pobieranie stron oszustów: wyłącznie HTML (bez JavaScriptu), limit rozmiaru i czasu, bez zapisywania plików wykonywalnych.
- Każda brakująca cecha → puste pole + wpis w `errors`; porównanie liczone na cechach dostępnych w obu materiałach.
- Materiałów NASK nie publikujemy: `data/` i `workspace/` są w `.gitignore`.

## Decyzje i kalibracja

Stan po integracji modułów (21.11, wieczór). Wszystkie liczby: tryb offline (`--offline`), bez LLM (`--no-llm`), CPU, zbiory z `python -m slad synthetic` (ziarno 7 = zbiór demonstracyjny). Dane NASK nie były jeszcze dostępne — wagi i progi wymagają potwierdzenia na nich (rozdz. „Kalibracja na danych NASK” niżej).

### Integracja (zmiany techniczne)

- `pyproject.toml`: `pythonpath = ["."]` dla pytest (testy działają bez instalacji pakietu); ruff ignoruje UP042 — enumy `(str, Enum)` w `slad/models.py` są kontraktem (StrEnum zmieniłby `str()` / `format()` wartości).
- `.gitignore`: pliki pochodne ingestu (`.slad_extracted/`), cache bez workspace (`.cache/`), cache galerii twarzy (`reference/faces/.gallery_cache.json` — embeddingi zdjęć referencyjnych), sekrety Streamlit. `.gitattributes`: `start.bat` z CRLF, `*.sh` z LF.
- `config/settings.yaml`: wszystkie rodzaje przesłanek jawnie w `link.weights` (wcześniej część brała wartości domyślne z kodu — strona „Metodyka” w panelu ich nie pokazywała). Wartości bez zmian.
- Panel → „Metodyka”: szkic `docs/METODYKA.md` (sam nagłówek) nie zastępuje już opisu wbudowanego; dokument uzupełniony pełnym opisem metody.
- Odchylenia od pierwotnego kontraktu (wszystkie zgodne wstecz): dodatkowe argumenty nazwane (`build(dates, progress)`, `process_dataset(force, ctx, stage_progress)`, `analyze_new(add, ctx)`, `load_dataset(label_from_dir)`), `process_dataset` zwraca podsumowanie zamiast `None`, brak `text_index.pkl` (kontekst w pamięci), dodatkowe pliki `run.json` / `report.json`. Szczegóły w tabeli modułów wyżej.

### Decyzje metodyczne podjęte przy integracji

1. **Porównanie tekstu kanałami (wejście / OCR, także na krzyż).** Test „nowy materiał spoza bazy” (warianty usunięte z bazy i analizowane od nowa) pokazał, że **sam tekst** reklamy nie był przypisywany do kampanii (0/7), a zrzut ekranu z lekko przyciętą kreacją dostawał wynik 3.99 < 4: tekst złączony (wejście + OCR + strona) pełnego rekordu „rozcieńczał” podobieństwo materiału częściowego (kosinus 0.6–0.87 dla tej samej treści). Dodano porównanie kanałami (`LinkContext.segment_sims`): najwyższe podobieństwo spośród par kanałów, waga liczona od długości dopasowanych kanałów. Kanały porównujemy na **zbiorach n-gramów znakowych 3–5 bez IDF** — z IDF jedna literówka w 90-znakowym tekście dawała kosinus 0.87 (unikalne n-gramy literówki mają najwyższą wagę, a szablon kampanii — niską). Wynik: sam tekst 0/7 → **7/7**, sam zrzut 8/8 (bez zmian), szumy 4/4 bez powiązań; zbiór ziarna 11: precyzja par 0.954 → 0.969, ARI 0.975 → 0.983.
2. **Sprawdzenie „wspólny szablon grafiki”** (podobny obraz przy zupełnie innym tekście → połowa wagi) zostaje na kosinusie TF-IDF treści reklamy, nie na nowej mierze kanałów: n-gramy bez IDF mają wyższe tło dla tekstów niepowiązanych (0.1–0.3), co wyłączało zabezpieczenie i dało fałszywe powiązanie (szum ↔ kampania, ziarno 11).
3. **Tekst niemal identyczny ≥ 60 znaków może potwierdzać powiązanie** (jak podobny tekst TF-IDF). Wcześniej reguła z dokumentacji („< 60 znaków nie jest przesłanką silną”) działała w praktyce od ~90 znaków, bo współczynnik długości był używany jako rzadkość. Waga nadal rośnie z długością (pełna od 120 znaków) — identyczne krótkie hasła wciąż nie łączą materiałów (test `test_short_identical_slogans_do_not_link`).
4. **Osoby spoza słownika** (heurystyka „imię + nazwisko” ze słownikiem imion i sprowadzeniem odmiany do mianownika) — tylko w treści reklamy (wejście + OCR), nie na stronie docelowej (fałszywe „opinie klientów”). Osoba pozostaje przesłanką miękką z limitem; trudny przypadek „ta sama osoba w dwóch kampaniach” (kampanie 01 i 02 + szum) nadal rozdzielony, a nazwy kampanii zawierają osobę („Marek Zawadzki · fałszywy wywiad · …”).
5. **Wagi i progi bez zmian** (`related` 4, `variant` 7, `assign.min_score` 4). Zbiór syntetyczny jest dla obecnej metody łatwy (wszystkie metryki 1.0), więc dalsze strojenie byłoby dopasowaniem do generatora — wagi kalibrujemy dopiero na danych NASK.
6. **LLM w przebiegach wsadowych wyłączony.** Lokalna Ollama (qwen3.5:9b, runner MLX) nie obsługuje `response_format`, więc klient przechodzi na JSON w treści promptu; czas: 8 materiałów w 105 s (≈ 13 s na materiał przy 4 wątkach), czyli ≈ 16 min dla 73 materiałów zbioru demonstracyjnego wobec 10 s bez LLM — powyżej limitu ~10 min. `python -m slad demo` ma więc LLM domyślnie wyłączony (`--llm` włącza). Przebiegi wsadowe i ewaluacja — `--no-llm` (heurystyki: słowniki, heurystyka osób, reguły narracji). LLM zalecany do analizy pojedynczego materiału albo z hostowanym endpointem (`.env`: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`); wyniki są w cache.

7. **Zgłoszenia z przeglądu modułów, poprawione przy integracji:** (a) ten sam plik pustego / jednolitego obrazu (zaślepka, pusty zrzut — bez pHash) w dwóch materiałach nie jest już przesłanką „identyczny plik graficzny” (wcześniej waga 5.0 omijała ochronę pustych obrazów i rzadkość); (b) witryny na kreatorach stron z listy wyjątków (`oszustwo.wordpress.com`, `promo.tilda.ws`) nie są traktowane jak sama platforma — zachowują domenę jako przesłankę i cechy strony (trackery, IP, TLS); (c) panel pokazuje miniatury i ramki twarzy z orientacją z EXIF (zdjęcia z telefonu), zgodnie z układem współrzędnych detekcji.

### Wyniki końcowe (offline, bez LLM)

| Zbiór / test | Wynik |
|---|---|
| ziarno 7 (demo): 73 mat., 8 kampanii + 12 szumów | pary P/R/F1 1.000 · B-Cubed 1.000 · ARI/NMI 1.000 · 212/212 trafnych powiązań (13 duplikatów, 173 warianty, 26 powiązanych) · przypisanie najbliższego sąsiada 61/61 · szumy 12/12 niepołączone · 0 materiałów z błędami · 10 s |
| ziarno 11: 97 mat., 10 kampanii + 25 szumów | pary P 0.969 / R 1.000 / F1 0.984 · ARI 0.983 · trafność powiązań 96.7% |
| ziarno 3: 101 mat., 10 kampanii + 25 szumów | pary P 0.950 / R 1.000 / F1 0.974 · ARI 0.973 · trafność powiązań 95.3% |
| nowe materiały spoza bazy (ziarno 7: 1 wariant z każdej kampanii ≥ 4 mat. + 4 szumy) | pełny rekord 8/8 · sam zrzut 8/8 · sam tekst 7/7 · sam URL offline 3/8 · szumy: 0 fałszywych przypisań |
| nowe materiały spoza bazy (ziarno 11) | pełny rekord 10/10 · sam zrzut 9/9 · sam tekst 7/9 · 1 szum przypisany (bliźniak, patrz niżej) |
| zbiór tylko ze zrzutów ekranu (69 mat.) | pary P 1.000 / R 0.841 · ARI 0.906 |
| zbiór tylko z tekstów (68 mat.) | pary P 1.000 / R 0.417 (kampanie z kilkoma szablonami tekstu dzielą się — bez wspólnego śladu to zamierzone) |
| zbiór tylko z URL, offline (66 mat.) | pary P 0.646 / R 0.844 — fikcyjny skracacz `krotko.example` (spoza listy skracaczy) wspólny dla kampanii 04 i 07 jest bez sieci nieodróżnialny od domeny docelowej |
| przerobiony zrzut znanej kreacji (kadr 3%, skala 0.8, jasność +12%, JPEG q70) | **przypisany** do właściwej kampanii (wynik 7.4: pHash Δ=6 + ta sama treść OCR) |
| niezwiązany tekst (przepis kulinarny) | **brak powiązań** |
| `analyze https://example.com` (sieć) | 4.4 s: przekierowania, DNS, ASN (AS13335), RDAP (rejestrator, data), TLS, tytuł, tekst strony |

Wszystkie błędy na ziarnach 11 i 3 to **bliźniacze szumy generatora**: przy `noise > 12` szablony szumów są używane ponownie (np. „Marek Zawadzki zaprasza na kurs gotowania online…” w dwóch materiałach z różnymi etykietami `noise-01` / `noise-13`) — teksty niemal identyczne, więc połączenie jest zgodne z metodą, a błędne tylko względem etykiet generatora.

### Kalibracja na danych NASK (do zrobienia po otrzymaniu zbioru)

1. `python -m slad ingest data/nask --show` — czy kolumny i obrazy zostały rozpoznane (`meta["ingest_warnings"]`).
2. `python -m slad run data/nask` (z siecią, jeśli strony żyją) → `python -m slad eval` (gdy są etykiety; kampanie zakodowane nazwami katalogów: `load_dataset(..., label_from_dir=True)`).
3. Raport `report.json → links.by_evidence`: rodzaje przesłanek o trafności < 90% → niższa waga w `config/settings.yaml`; lista `errors.false_links` / `merged_campaigns` pokazuje, która przesłanka skleja kampanie.
4. Próg `link.thresholds.related` i `link.assign.min_score` — wg krzywej precyzja/pełność par.
