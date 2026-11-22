# Metodyka — Ślad Fejka

> Jak z pojedynczych podejrzanych reklam powstają kampanie: jakie cechy pozyskujemy i skąd, jak liczymy podobieństwo dwóch materiałów, jak grupujemy je w kampanie i jak przypisujemy nowy materiał.
> Każde powiązanie ma listę **przesłanek** (co jest wspólne i z jaką wagą) — wynik jest zawsze wyjaśnialny.
> Architektura i kontrakty modułów: [ARCHITEKTURA.md](ARCHITEKTURA.md). Wagi i progi: [`config/settings.yaml`](../config/settings.yaml).

## 1. Założenia

- **Materiał** to cokolwiek, co analityk dostaje: URL reklamy lub strony, zrzut ekranu, zapisany HTML, sam tekst albo rekord zbioru (wiele pól naraz). Każdy materiał może być **niekompletny** — brak obrazu, martwa strona, brak metadanych. Porównujemy wyłącznie cechy dostępne w obu materiałach; brak cechy to nie błąd.
- **Kampania** = materiały jednego operatora / jednej akcji. Łączą je ślady operatora (identyfikatory, infrastruktura, ta sama kreacja i treść), a **nie** sam motyw: ta sama twarz celebryty, ta sama narracja („inwestycja w krypto”) czy wspólny Cloudflare występują w wielu niezależnych kampaniach i same nie mogą ich połączyć.
- **Precyzja przed pełnością.** Fałszywe połączenie dwóch kampanii jest dla analityka groźniejsze niż niepołączony wariant — dlatego słabe sygnały wymagają potwierdzenia, a częste wartości tracą wagę.
- Etykiety kampanii ze zbioru służą **wyłącznie ewaluacji** — żaden moduł cech ani powiązań ich nie czyta.

```
 materiał ─► CECHY: tekst · obraz · infrastruktura ─► PRZESŁANKI dla par (wagi × rzadkość)
          ─► POWIĄZANIA (duplikat / wariant / powiązany) ─► KAMPANIE (graf + Louvain)
          ─► nowy materiał: przypisanie / nowa kampania / brak powiązań — zawsze z uzasadnieniem
```

## 2. Pozyskiwanie danych i cech

### 2.1 Wejście (`slad/ingest`)

Format nie jest znany z góry, więc wczytywanie rozpoznaje go samo: JSON/JSONL, CSV/TSV (separator i kodowanie, także cp1250 z polskiego Excela), XLSX (wszystkie arkusze, scalone komórki, hiperłącza, osadzone obrazy), katalogi zrzutów ekranu i stron HTML/MHTML, pliki EML, listy URL. Kolumny są mapowane po synonimach PL/EN („URL obrazu”, „Nazwa kampanii”, pola Biblioteki Reklam Meta) i po treści (adresy, ścieżki obrazów, długi tekst). Zdefangowane adresy (`hxxps://scam[.]pl`) są przywracane, daty normalizowane, problemy opisywane po polsku w `meta["ingest_warnings"]` — zły rekord jest pomijany z ostrzeżeniem, nigdy nie przerywa wczytywania. `python -m slad ingest <zbiór> --show` pokazuje, jak zbiór został odczytany.

### 2.2 Treść (`slad/features/text.py`, `ocr.py`, `iocs.py`, `llm.py`)

| Cecha | Metoda | Po co |
|---|---|---|
| tekst zrzutów | OCR RapidOCR (PP-OCR, ONNX, CPU); wysokie zrzuty cięte na pasy zamiast zmniejszania | treść reklamy podanej jako obraz |
| normalizacja | NFKC, znaki zerowej szerokości, homoglify (cyrylica w łacińskich słowach), cudzysłowy | odporność na zaciemnianie tekstu |
| **IOC** | telefony (E.164; odrzucane daty, kwoty, PESEL), e-maile, URL/domeny, portfele BTC/ETH/TRX (sumy kontrolne Base58Check/bech32), Telegram, WhatsApp, IBAN (mod-97) | **identyfikatory operatora** — najsilniejsze przesłanki |
| osoby | słownik osób publicznych z odmianą (`config/gazetteer/persons.yaml`) + heurystyka „imię + nazwisko” w treści reklamy (`first_names.yaml`, sprowadzenie do mianownika: „Markiem Zawadzkim” → „Marek Zawadzki”) + opcjonalnie LLM | podszywanie się pod osobę |
| marki | słownik marek, mediów i instytucji (`brands.yaml`), skróty wielkimi literami (PKO, ZUS) | podszywana marka |
| narracja | 13 reguł (`config/narratives.yaml`): inwestycja w krypto, bot AI, fałszywy wywiad, dopłata do przesyłki, program rządowy, cudowny lek… | motyw oszustwa (przesłanka miękka) |
| obietnica / CTA / produkt | wyrażenia regularne (+ LLM, gdy dostępny) | opis kampanii w UI |
| słowa kluczowe | bez stopwords, formy odmiany grupowane | przesłanka miękka, nazwa kampanii |
| MinHash | 128 permutacji, szyngle 3 słów (datasketch) | teksty niemal identyczne, LSH |

LLM (dowolne API zgodne z OpenAI, np. lokalna Ollama) jest **opcjonalny**: uzupełnia osoby, marki, produkt i narrację w schemacie JSON; bez LLM działają heurystyki. Wyniki LLM są sprowadzane do nazw ze słownika i do dozwolonych tagów narracji — model nie może wprowadzić dowolnych etykiet.

### 2.3 Obraz (`slad/features/visual.py`, `faces.py`, `clip.py`)

- **Hashe percepcyjne** pHash / dHash / wHash (64 bity) i colorHash na kopii roboczej ≤ 256 px **bez jednolitych ramek** — ta sama kreacja zmniejszona, skompresowana, w ramce lub lekko przycięta ma bliski hash (pomiar: JPEG q60 Δ=0, połowa rozmiaru Δ=2, jasność +15% Δ≤6, kadr 3% Δ≤8; różne obrazy: min. Δ=20). Obrazy puste/jednolite nie dostają hashy (nie mogą „być duplikatami” siebie nawzajem).
- **Twarze**: detekcja YuNet + embedding SFace (OpenCV, 128-d). Bramka jakości: twarze rysowane i nieczytelne (mała norma wektora) nie dostają embeddingu. Tożsamość z bazy referencyjnej `reference/faces/<Imię_Nazwisko>/` (kosinus ≥ 0.363, próg zalecany dla SFace) trafia do osób materiału.
- **CLIP** (opcjonalnie, `requirements-extra.txt`) — podobieństwo semantyczne grafik („ten sam układ”).

### 2.4 Infrastruktura (`slad/features/infra/`)

| Źródło | Cechy | Znaczenie |
|---|---|---|
| łańcuch przekierowań (HTTP 30x, meta refresh, proste JS, nagłówek `Refresh`) | adresy pośrednie, strona docelowa | skracacze, cloakery, TDS → domena docelowa |
| HTML strony | tytuł, **identyfikatory trackerów** (GA/GA4, GTM, Facebook Pixel, Yandex Metrica, TikTok, Hotjar, Clarity…), tokeny weryfikacji domen, ukryte pola ofert CPA, formularze i adresy POST, domeny skryptów, **simhash DOM** (szkielet strony bez tekstu) | konto operatora, zestaw „kit”, szablon |
| favicona | hash mmh3 (styl Shodana) | ten sam panel / szablon |
| DNS | A/AAAA, CNAME, NS, MX | hosting |
| IP → ASN | Team Cymru (DNS), RDAP IP | sieć hostingu (słaba przesłanka) |
| RDAP domeny | rejestrator, data rejestracji | domeny rejestrowane hurtowo |
| TLS | odcisk certyfikatu, SAN, wystawca | wspólny certyfikat = wspólny operator |
| crt.sh (opcja) | inne domeny z Certificate Transparency | rotowane domeny kampanii |

Bezpieczeństwo i odporność: pobieramy wyłącznie HTML (bez JavaScriptu i plików wykonywalnych), z limitem rozmiaru i czasu; nie odpytujemy adresów prywatnych ani domen zarezerwowanych; nie pobieramy stron platform (Facebook, TikTok…) — ich szablon i favicona połączyłyby wszystko; rozwijamy przekierowania platform (`l.facebook.com/l.php?u=`). **Strony generyczne** (HTTP ≥ 400, parking domeny, zawieszone konto, challenge Cloudflare, ostrzeżenie o phishingu) nie dają cech treści — ich szablon należy do hostingu, nie do operatora. Wszystkie odpowiedzi są w pamięci podręcznej (`workspace/<nazwa>/cache/`), więc powtórne uruchomienie działa offline. Gotowe dane infrastruktury z rekordu (`meta.infra`) i zapisane strony mają pierwszeństwo przed pobraniem na żywo.

## 3. Podobieństwo dwóch materiałów — przesłanki

### 3.1 Wynik

Wynik pary to **suma wag przesłanek**; waga ≈ log-iloraz szans „ta sama kampania” (im rzadszy i bardziej osobisty ślad, tym wyższa). Każda przesłanka = rodzaj, grupa (tekst / obraz / osoby / infrastruktura / narracja), opis po polsku, wspólna wartość, podobieństwo i wkład w wynik. Suma wkładów zawsze równa się wynikowi.

| Poziom | Rodzaje (waga bazowa) | Zasada |
|---|---|---|
| **silne** — identyfikatory operatora i ta sama treść | portfel krypto 7 · tracker / piksel 6 · telefon 6 · e-mail 6 · IBAN 6 · Telegram 5 · WhatsApp 5 · certyfikat TLS 5 · domena docelowa 5 · skrócony link 5 · domena z certyfikatu 4 · tekst niemal identyczny 5 · ta sama grafika (pHash Δ ≤ 6) 5 · identyczny HTML | jedna wystarcza do powiązania |
| **średnie** | favicona 4 · adres formularza 4 · host 3 · domena pośrednia 3 · domena skryptów 3 · ID oferty CPA 3 · szablon DOM 3 · podobny tekst 3 · IP 2.5 · tytuł 2 · zestaw pól formularza 2 | wymagają przesłanki silnej albo ≥ 2 różnych rodzajów średnich |
| **miękkie** | twarz 4 · CLIP 3 · osoba 2 · rejestrator + data ±3 dni 1.5 · serwer DNS 1.5 · marka 1 · narracja 1 · ASN 0.5 · słowa kluczowe ≤ 1 | nigdy same nie łączą; łącznie ograniczone |

Wagi cech ciągłych zależą od podobieństwa (np. pHash: 5 × (1 − Δ/12); tekst: od progu podobieństwa do 1), a przesłanek tekstowych — także od długości krótszego tekstu.

### 3.2 Ważenie rzadkością

Wartość wspólna dla dużej części zbioru to „tło internetu” (Cloudflare, popularny DNS, domyślna favicona hostingu, skracacz), a nie ślad operatora. Waga przesłanki jest mnożona przez współczynnik rzadkości `r = (ln(1 + n/df) / ln(1 + n/free))²` (min. 0.05; `r = 1` dla `df ≤ free`), gdzie `df` — liczba materiałów z tą wartością, `n` — wielkość zbioru, `free` — próg bez kary zależny od rodzaju: identyfikatory operatora do 35% zbioru (ich częstość odzwierciedla wielkość kampanii), domena docelowa do 25%, pozostałe do 10% (maks. 50 materiałów). Dla cech ciągłych (obraz, szablon DOM) rzadkość wynika z **gęstości sąsiedztwa**: ile innych materiałów ma obraz / szablon w promieniu Δ ≤ 6 — wspólny układ zrzutów ekranu platformy waży mało. Niezależnie od statystyk listy `config/stoplists.yaml` usuwają wartości zawsze częste (CDN-y, wielkie platformy, publiczne skracacze, wystawcy certyfikatów, standardowe pola formularzy).

### 3.3 Reguły ochronne (przeciw fałszywym połączeniom)

- **Potwierdzenie:** bez przesłanki silnej (o rzadkości ≥ 0.75) i przy < 2 rodzajach przesłanek średnich wynik jest obcinany poniżej progu — wspólna favicona hostingu + wspólny skracacz to wciąż tło.
- **Limit miękkich:** suma przesłanek miękkich ≤ min(0.875 × próg, max(0.5 × próg, 2 × suma twardych)).
- **Osoby:** nazwisko + twarz łącznie ≤ 0.75 × próg — **ta sama twarz celebryty w różnych kampaniach ich nie łączy**.
- **Krótkie teksty** (< 60 znaków) nie są przesłanką silną, a waga przesłanek tekstowych rośnie z długością (pełna od 120 znaków) — szablonowe hasła („Kliknij i sprawdź!”) bywają identyczne w niezależnych reklamach.
- **Wspólny szablon grafiki:** podobny obraz przy zupełnie innym tekście reklamy (kosinus < 0.15) to raczej wspólny szablon niż ta sama kreacja — połowa wagi, przesłanka średnia.
- **Role domen:** domena widziana w korpusie jako pośrednia (skracacz, cloaker, TDS) jest przesłanką średnią także w materiałach z martwym łańcuchem; IP materiałów stojących wyłącznie za CDN są pomijane; subdomeny darmowych hostingów (`*.netlify.app`, `*.wixsite.com`) są osobnymi witrynami.

### 3.4 Tekst

Trzy uzupełniające się miary: (1) Jaccard sygnatur **MinHash** (szyngle 3 słów) — tekst niemal identyczny; (2) kosinus **TF-IDF n-gramów znakowych 3–5** całego tekstu i samej treści reklamy (wejście + OCR, bez strony docelowej) — podobna treść odporna na odmianę i błędy OCR; (3) **porównanie kanałami** — tekst z wejścia i tekst z OCR każdego materiału osobno, także na krzyż, na zbiorach n-gramów znakowych (bez IDF: unikalne n-gramy literówki nie mogą zaniżać podobieństwa kopii tej samej treści). Dzięki (3) ten sam tekst reklamy podany raz jako pole tekstowe, a raz tylko na zrzucie ekranu, albo sam zrzut porównany z pełnym rekordem (tekst + zrzut + strona), wciąż daje przesłankę „treść reklamy niemal identyczna — w jednym materiale tekst, w drugim napis na obrazie (OCR)”.

### 3.5 Relacja

- **duplikat** — treść niemal identyczna we wszystkich grupach treści obecnych w obu materiałach (tekst: Jaccard ≥ 0.9 lub kosinus ≥ 0.98; obraz: ten sam plik albo pHash i dHash Δ ≤ 2);
- **wariant** — wynik ≥ 7 (np. ta sama kampania, inna kreacja / domena z rotacji);
- **powiązany** — wynik ≥ 4 (próg powiązania).

Wysoki wynik z samej wspólnej infrastruktury to wariant, nie duplikat.

### 3.6 Skalowalność — kandydaci

Do 400 materiałów porównujemy wszystkie pary. Powyżej — tylko kandydatów z: indeksu odwrotnego wskaźników (z pominięciem wartości bardzo częstych), MinHash LSH, dokładnego wyszukiwania Hamminga pHash ≤ 10 i simhash DOM ≤ 6 (numpy, popcount), najbliższych twarzy i sąsiadów TF-IDF. Pomiar: budowa powiązań dla 3 000 materiałów ≈ 3 s, 10 000 ≈ 9 s (CPU); analiza nowego materiału na bazie 3 000 — ≈ 20 ms po pierwszym wczytaniu. Najdroższy etap to OCR zrzutów (wykonywany raz, wynik w cache).

## 4. Grupowanie w kampanie (`slad/link/cluster.py`)

1. Graf: węzły = materiały, krawędzie = powiązania ≥ próg „powiązany” (waga = wynik).
2. Spójne składowe to kandydaci na kampanie.
3. Składowe > 6 węzłów są dzielone algorytmem **Louvain** (deterministycznie) — przeciw „łańcuchowaniu”, gdy jedno słabe powiązanie skleiłoby dwie kampanie; duplikaty nigdy nie są rozdzielane.
4. Części są ponownie łączone, gdy łączy je gęsta sieć powiązań (≥ 30% par) albo ≥ 2 silne powiązania — Louvain nie może rozbić wariantów jednej kampanii; pojedyncze materiały wracają do swojej części.
5. Kampania dostaje nazwę po polsku (osoba lub marka · narracja · domena docelowa), spójność (średni wynik powiązań), najczęstsze osoby / marki / narracje / domeny / wskaźniki, okres aktywności i **kluczowe przesłanki**. Identyfikatory kampanii są stabilne między przeliczeniami (K003 pozostaje K003 po dodaniu materiału).

Opcjonalnie `link.cluster.method: hdbscan` (odległość 1/(1 + wynik)); na danych testowych dzieli kampanie częściej niż metoda grafowa, dlatego domyślna jest metoda grafowa.

## 5. Nowy materiał (`slad/link/assign.py`)

Cechy nowego materiału są porównywane z kandydatami z bazy. Wynik kampanii = najlepsze powiązanie z jej materiałem + 0.25 × drugie najlepsze (kilka zgodnych materiałów wzmacnia przypisanie). Wynik ≥ 4 → **przypisany**; w przeciwnym razie powiązania z materiałami spoza kampanii → **potencjalnie nowa kampania**; inaczej **brak powiązań** — wtedy uzasadnienie wskazuje najbliższy materiał i mówi, czego zabrakło („ta sama narracja i marka — to za mało, by uznać materiały za powiązane”). Opcja „dodaj do bazy” przelicza kampanie z zachowaniem identyfikatorów.

## 6. Interpretowalność

- Każde powiązanie: lista przesłanek z wagą i opisem po polsku, np. *„wspólny identyfikator Facebook Pixel 731372159010928 (+6.0)”*, *„ta sama lub przerobiona grafika (pHash Δ=4) (+3.3)”*, *„ta sama narracja (Inwestycja w kryptowaluty) — wartość częsta (10 mat. w zbiorze), waga obniżona (+0.8)”*.
- Kampania: „Dlaczego to jedna kampania?” — udział grup przesłanek i najczęstsze przesłanki wewnątrz kampanii.
- Graf powiązań z trybem „podążaj za śladem” (węzły wspólnych wskaźników: tracker, domena, IP, portfel…).
- Eksport do Neo4j (CSV + Cypher), GraphML (Gephi) i JSON — przesłanki zachowane na krawędziach.

## 7. Odporność na niekompletne dane, duplikaty i różnice formatu

| Sytuacja | Zachowanie |
|---|---|
| brak obrazu / tekstu / URL | porównanie na cechach dostępnych w obu materiałach; `coverage` powiązania pokazuje część wspólnych grup |
| martwa strona, błąd DNS, 404 | wpis w `errors`, strona generyczna bez cech treści; domena i łańcuch z wejścia nadal porównywane |
| sam zrzut vs pełny rekord | OCR + pHash + porównanie kanałami treści |
| dokładny duplikat (repost) | relacja „duplikat”; duplikaty nigdy nie są rozdzielane przy grupowaniu |
| różne formaty zbioru | automatyczne rozpoznanie formatu i kolumn (rozdz. 2.1) |
| brak sieci | `--offline`: dane z rekordu, zapisane strony i cache |

## 8. Ewaluacja i kalibracja

Do czasu otrzymania danych NASK metoda jest sprawdzana na **syntetycznym zbiorze z etykietami** (`python -m slad synthetic`): 8 fikcyjnych kampanii (4–9 wariantów) + 12 materiałów-szumów, z celowo trudnymi przypadkami — ta sama osoba w dwóch kampaniach, wspólny Cloudflare / serwery DNS / rejestrator / skracacz / domyślna favicona, ~20% rekordów bez obrazu, URL lub tekstu, martwe strony, duplikaty, kampanie wspólne tylko narracją. Metryki (`python -m slad eval`): precyzja / pełność / F1 par, B-Cubed, ARI/NMI, trafność powiązań wg relacji i rodzaju przesłanki, przypisanie najbliższego sąsiada, „nie łącz niepowiązanych”.

Wyniki (offline, bez LLM; szczegóły i decyzje kalibracyjne: [ARCHITEKTURA.md → Decyzje i kalibracja](ARCHITEKTURA.md#decyzje-i-kalibracja)):

| Test | Wynik |
|---|---|
| zbiór syntetyczny (73 mat.) | F1 par 1.000 · B-Cubed 1.000 · ARI 1.000 · 212/212 trafnych powiązań · 12/12 szumów niepołączonych |
| inne ziarno, 10 kampanii + 25 szumów (97 mat.) | precyzja par 0.969 · pełność 1.000 · ARI 0.983 (wszystkie błędy: szumy o niemal identycznym tekście) |
| **nowe materiały spoza bazy** (warianty usunięte z bazy i analizowane od nowa) | pełny rekord 8/8 · sam zrzut ekranu 8/8 · sam tekst 7/7 · szumy 4/4 bez powiązań |
| zbiór z samymi zrzutami ekranu | precyzja par 1.000 · pełność 0.84 · ARI 0.91 |

Kalibracja na danych NASK: raport ewaluacji podaje trafność **każdego rodzaju przesłanki** — rodzaje o niskiej trafności dostają niższą wagę w `config/settings.yaml` (bez zmian w kodzie); progi relacji i przypisania również są w konfiguracji.

## 9. Ograniczenia

- Same adresy URL bez sieci: nieznany (spoza list) skracacz współdzielony przez dwie kampanie wygląda jak wspólna domena docelowa. Z siecią łańcuch przekierowań rozstrzyga rolę domeny.
- Teksty jednej kampanii pisane różnymi szablonami nie łączą się bez wspólnego śladu (osoba i narracja są celowo zbyt słabe) — kampania może zostać podzielona na kilka grup.
- pHash nie łączy przeróbek zmieniających układ (kwadrat vs baner); pomaga opcjonalny CLIP.
- Wagi i progi skalibrowane na danych syntetycznych — wymagają potwierdzenia na zbiorze NASK.
