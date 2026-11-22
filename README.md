# Ślad Fejka

**Od pojedynczych reklam do kampanii oszustw.** Narzędzie, które z pojedynczych podejrzanych reklam wyciąga wspólne ślady — treść, obraz i infrastrukturę — i łączy pozornie niezależne materiały w kampanie oszustw.

Zadanie konkursowe NASK **„Kolekcjoner Fejków — Follow the Fake”**, HackNation 2026 (Bydgoszcz, 21–22.11.2026). Zespół **zschie**.

> Status: **działający demonstrator end-to-end** (wczytanie → cechy → powiązania z przesłankami → kampanie → panel, analiza nowego materiału, eksport Neo4j, Docker). Sprawdzony na syntetycznym zbiorze z etykietami; kalibracja wag na przykładowym zbiorze NASK — gdy tylko trafi do `data/nask/`.

## Problem

Fałszywe reklamy z wizerunkami znanych osób są publikowane masowo i prowadzą do spreparowanych stron na krótko żyjących domenach. Pojedyncze zgłoszenie ma niewielką wartość analityczną — wzorce widać dopiero po zestawieniu wielu przypadków: wspólnej narracji, tych samych twarzy i szablonów graficznych, domen i przekierowań.

## Jak to działa

```
materiał: URL reklamy/strony · zrzut ekranu · rekord ze zbioru NASK
  ─► POZYSKANIE    pobranie strony, łańcuch przekierowań, zrzut, OCR
  ─► CECHY         treść i narracja: OCR + LLM (osoba, produkt, obietnica zysku, wezwanie do działania)
                   obraz: perceptual hash, embeddingi, wspólne szablony i logotypy
                   infrastruktura: domena, WHOIS/RDAP, DNS, IP/ASN, przekierowania, struktura strony
  ─► PODOBIEŃSTWO  ważone per grupa cech, odporne na brakujące dane
  ─► GRUPOWANIE    graf powiązań → kampanie; wykrywanie duplikatów i wariantów
  ─► PREZENTACJA   panel webowy: graf kampanii + przesłanki każdego połączenia
```

- **Nowy materiał** jest porównywany z bazą znanych kampanii → przypisanie do kampanii albo sygnał „potencjalnie nowa kampania”.
- **Interpretowalność:** każde powiązanie pokazuje, które cechy są wspólne i z jaką wagą.
- **Niekompletne dane** (martwa domena, brak metadanych, część materiału) obniżają pewność, ale nie blokują analizy.
- OCR i ekstrakcję cech przez LLM ze schematem JSON przenosimy z naszego drugiego projektu (Odwołania2JSON, UZP).

## Szybki start

Wymagania: Python 3.13 (CPU, bez GPU) albo Docker. Pierwsze uruchomienie skryptu tworzy `.venv`, instaluje zależności (`requirements.txt`) i pobiera modele twarzy (≈ 40 MB, `python -m slad models`).

```bash
./start.sh demo                            # zbiór syntetyczny → przetwarzanie offline → ewaluacja (≈ 30 s)
./start.sh ui --workspace workspace/demo   # panel: http://localhost:8501
```

Windows: `start.bat demo`, `start.bat ui --workspace workspace/demo`. Docker:

```bash
docker compose up --build app                          # panel: http://localhost:8501
docker compose run --rm demo                           # demo w kontenerze → ./workspace/demo
SLAD_WORKSPACE=workspace/demo docker compose up app    # panel z wynikami demo
DATASET=data/nask docker compose run --rm pipeline     # przetworzenie zbioru z ./data
```

### Praca ze zbiorem (np. NASK w `data/nask/`)

```bash
python -m slad ingest data/nask --show     # jak rozpoznano format, kolumny, obrazy (bez ekstrakcji cech)
python -m slad run data/nask               # cechy (z siecią; --offline bez niej) → powiązania → kampanie
python -m slad campaigns                   # kampanie w terminalu (python -m slad campaigns K001 — szczegóły)
python -m slad analyze zrzut.png           # nowy materiał: URL / zrzut ekranu / zapisany HTML / "tekst"
python -m slad export --format neo4j       # Neo4j (CSV + import.cypher), także graphml / json / all
python -m slad eval                        # metryki, gdy zbiór ma etykiety kampanii
python -m slad ui                          # panel webowy
```

(`python` = `.venv/bin/python` albo `./start.sh <komenda>`.) Wszystkie komendy: `python -m slad --help`.

### Konfiguracja

- `config/settings.yaml` — wagi przesłanek, progi relacji i przypisania, sieć, OCR, twarze; `python -m slad link` przelicza powiązania po zmianie wag bez ponownej ekstrakcji cech.
- `.env` (kopia `.env.example`) — opcjonalny LLM zgodny z API OpenAI (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`; np. lokalna Ollama), `SLAD_OFFLINE=1`, `SLAD_WORKSPACE`. Bez LLM działają heurystyki.
- `reference/faces/<Imię_Nazwisko>/*.jpg` — opcjonalna baza twarzy osób, pod które podszywają się reklamy (tylko lokalnie).
- `config/stoplists.yaml`, `config/narratives.yaml`, `config/gazetteer/` — listy wyjątków, reguły narracji, słowniki osób i marek.

### Dokumentacja

- [docs/METODYKA.md](docs/METODYKA.md) — pozyskiwanie cech, podobieństwo, grupowanie, przypisanie, ewaluacja.
- [docs/ARCHITEKTURA.md](docs/ARCHITEKTURA.md) — moduły i kontrakty, katalog roboczy, decyzje i kalibracja.
- [docs/team/ROADMAP_RU.md](docs/team/ROADMAP_RU.md) — co zrobione, co zostało (dla zespołu).

Testy: `.venv/bin/pytest -q` (bez sieci), styl: `.venv/bin/ruff check .`.

## Plan do niedzieli (22.11, 11:00)

- [x] wczytanie zbiorów w dowolnym formacie i model danych materiału
- [x] ekstrakcja cech: treść (OCR + heurystyki / LLM), obraz (hashe, twarze), infrastruktura (przekierowania, HTML, DNS, RDAP, TLS)
- [x] miara podobieństwa z przesłankami i grupowanie w kampanie, przypisanie nowego materiału
- [x] panel webowy (Streamlit) z grafem kampanii i przesłankami powiązań
- [x] opis metodyki (`docs/METODYKA.md`)
- [x] Docker + instrukcja instalacji i uruchomienia
- [ ] wczytanie przykładowego zbioru NASK, kalibracja wag, raport jakości
- [ ] prezentacja / wideo dla jury

## Wymagania NASK (checklista zgłoszenia)

- działający demonstrator z prostym interfejsem graficznym — pełny proces end-to-end,
- opis metodyki: pozyskiwanie cech, obliczanie podobieństwa, grupowanie,
- instrukcja instalacji, konfiguracji i uruchomienia wraz z zależnościami,
- kod źródłowy w repozytorium Git; mile widziana wersja skonteneryzowana.

**Kryteria oceny:** jakość korelacji i grupowania 30% · użyteczność i trafność cech oraz źródeł danych 30% · kompletność demonstratora end-to-end 15% · czytelność i interpretowalność powiązań 15% · potencjał wdrożeniowy, skalowalność i przenośność 10%.

## Dane

Przykładowy zbiór reklam typu scam udostępnia NASK (kanał Discord HackNation, [science.nask.pl/hacknation](https://science.nask.pl/hacknation)). **Materiałów organizatora nie wolno publicznie rozpowszechniać** — trzymamy je lokalnie w `data/` (wykluczone w `.gitignore`), nigdy w repozytorium. Dodatkowe dane z publicznych źródeł opisujemy z podaniem pochodzenia.

## Linki

- Materiały NASK: https://science.nask.pl/hacknation
- HackNation 2026: https://hacknation.pl

## Zespół

Zespół *zschie* — HackNation 2026.
