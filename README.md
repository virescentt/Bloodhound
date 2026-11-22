<p align="center"><img src="slad/ui/assets/bloodhound.png" width="140" alt="Bloodhound — logo"></p>

# Bloodhound

Łączy pojedyncze fałszywe reklamy w kampanie oszustw: wyciąga wspólne ślady w treści, obrazie i infrastrukturze stron i pokazuje, dlaczego materiały zostały połączone. Zadanie konkursowe NASK „Kolekcjoner Fejków — Follow the Fake”, HackNation 2026.

## Dokumentacja

- [Metodyka](docs/METODYKA.md) — cechy, podobieństwo, grupowanie, przypisanie nowego materiału, ewaluacja.
- [Architektura](docs/ARCHITEKTURA.md) — moduły, katalog roboczy, decyzje i wyniki.

## Możliwości

- Wejście: URL reklamy lub strony, zrzut ekranu, zapisany HTML/MHTML, plik HAR, tekst albo cały zbiór (CSV, XLSX, JSON, katalogi).
- Cechy: treść (OCR, telefony, e-maile, portfele krypto, Telegram, IBAN, osoby, marki, narracje), obraz (hashe percepcyjne, twarze), infrastruktura (przekierowania, trackery GA/GTM/Pixel, szablon strony, favicon, DNS, RDAP, IP/ASN, TLS).
- Każde powiązanie ma przesłanki z wagami; wartości częste (np. Cloudflare, popularny hosting) ważone rzadkością.
- Kampanie na grafie powiązań; nowy materiał → znana kampania, nowa kampania albo brak powiązań.
- Odporność na niekompletne dane i duplikaty; strony oszustów pobierane bez JavaScriptu, z blokadą adresów wewnętrznych.
- Panel webowy z grafem, eksport do Neo4j / GraphML / JSON, Docker. Działa na CPU, LLM opcjonalny.

## Instalacja i uruchomienie

Wymagany **Python 3.13** albo Docker. Pierwsze uruchomienie skryptu tworzy `.venv`, instaluje zależności i pobiera modele twarzy (≈ 40 MB).

**macOS / Linux**
```bash
./start.sh demo                            # zbiór przykładowy (syntetyczny) → kampanie → ewaluacja
./start.sh ui --workspace workspace/demo   # panel → http://localhost:8501
```

**Windows**
```bat
start.bat demo
start.bat ui --workspace workspace/demo
```

**Docker**
```bash
docker compose run --rm demo                          # demo → ./workspace/demo
SLAD_WORKSPACE=workspace/demo docker compose up app   # panel → http://localhost:8501
```

**Własny zbiór**, np. materiały NASK w `data/nask/` (folder `data/` nie trafia do repozytorium):

```bash
./start.sh run data/nask   # cechy → powiązania → kampanie
./start.sh                 # panel → http://localhost:8501
```

Pozostałe komendy (analiza pojedynczego materiału, eksport, ewaluacja): `./start.sh --help`.

**LLM (opcjonalnie)** — w pliku `.env` (kopia `.env.example`) dowolne API zgodne z OpenAI, np. Qwen3.5-9B na OVHcloud AI Endpoints. Bez klucza działają reguły i słowniki, także offline.

## Zespół

Zespół *Wanna Pizza* — HackNation 2026.
