# Ślad Fejka

**Od pojedynczych reklam do kampanii oszustw.** Narzędzie, które z pojedynczych podejrzanych reklam wyciąga wspólne ślady — treść, obraz i infrastrukturę — i łączy pozornie niezależne materiały w kampanie oszustw.

Zadanie konkursowe NASK **„Kolekcjoner Fejków — Follow the Fake”**, HackNation 2026 (Bydgoszcz, 21–22.11.2026). Zespół **zschie**.

> Status: **w budowie** — repozytorium robocze; demonstrator powstaje w trakcie hackathonu.

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

## Plan do niedzieli (22.11, 11:00)

- [ ] wczytanie przykładowego zbioru NASK i model danych materiału
- [ ] ekstrakcja cech: treść (OCR + LLM), obraz, infrastruktura
- [ ] miara podobieństwa i grupowanie w kampanie
- [ ] panel webowy (Streamlit / Gradio) z grafem kampanii i przesłankami powiązań
- [ ] opis metodyki (`docs/METODYKA.md`)
- [ ] Docker + instrukcja instalacji i uruchomienia

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
