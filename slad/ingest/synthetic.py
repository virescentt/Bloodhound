"""Syntetyczny zbiór reklam-oszustw z etykietami kampanii — testy i demo bez danych NASK.

    generate(out_dir, seed=7, n_campaigns=8, noise=12) -> Path   # ścieżka do records.jsonl

Wynik (deterministyczny dla danego ``seed``):

    out_dir/records.jsonl   pola: id, url, text, image, html, platform, advertiser, date, campaign, infra
    out_dir/images/*.png    kreacje reklam (układ, kolory i nagłówek kampanii + sylwetka osoby)
    out_dir/pages/*.html    strony docelowe (wspólny szkielet DOM kampanii, trackery, formularz, favicona)
    out_dir/README.md       opis zbioru (PL)

Wszystkie osoby, marki, produkty i wskaźniki są fikcyjne. Infrastruktura wyłącznie w przestrzeniach
zarezerwowanych: domeny .test / .example, adresy IP z TEST-NET (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24),
ASN z puli dokumentacyjnej (wyjątek: wspólny „AS13335 CLOUDFLARENET” — celowo, jako częsta, słaba przesłanka).
Pole ``infra`` zawiera gotowe dane sieciowe (klucze = pola InfraFeatures), więc potok działa offline.

Trudne przypadki (jak w ukrytym zbiorze NASK):
  (a) dwie różne kampanie podszywają się pod tę samą osobę,
  (b) wspólna infrastruktura kilku kampanii (Cloudflare, popularne serwery nazw i rejestrator, skracacz linków,
      domyślna favicona hostingu, generator polityki prywatności) — sama w sobie nie może łączyć,
  (c) ~20% rekordów bez obrazu / adresu / tekstu, martwe strony (brak HTML, infrastruktura z błędami),
  (d) dokładne duplikaty (ponownie opublikowana ta sama grafika i tekst),
  (e) pojedyncze materiały-szum z unikalnymi etykietami noise-01, noise-02, …,
  (f) kampanie wspólne wyłącznie narracją.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import random
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

log = logging.getLogger(__name__)

W, H = 600, 400

# ---------------------------------------------------------------- fikcyjne osoby


@dataclass(frozen=True)
class Persona:
    nom: str  # mianownik
    gen: str  # dopełniacz
    ins: str  # narzędnik („wywiad z …”)
    role: str
    skin: str
    hair: str
    suit: str


PERSONAS: dict[str, Persona] = {
    "zawadzki": Persona(
        "Marek Zawadzki",
        "Marka Zawadzkiego",
        "Markiem Zawadzkim",
        "prezenter telewizyjny",
        "#e8b98f",
        "#3b2a1e",
        "#1f2a44",
    ),
    "wierzbicka": Persona(
        "Joanna Wierzbicka", "Joanny Wierzbickiej", "Joanną Wierzbicką", "ekonomistka", "#f1c7a3", "#a0522d", "#5b2333"
    ),
    "domanski": Persona(
        "Piotr Domański",
        "Piotra Domańskiego",
        "Piotrem Domańskim",
        "wiceminister finansów",
        "#e0ac84",
        "#9e9e9e",
        "#22313f",
    ),
    "brzezinski": Persona(
        "Tomasz Brzeziński",
        "Tomasza Brzezińskiego",
        "Tomaszem Brzezińskim",
        "przedsiębiorca technologiczny",
        "#d9a47c",
        "#1b1b1b",
        "#2e2e38",
    ),
    "nowicka": Persona(
        "Ewa Nowicka", "Ewy Nowickiej", "Ewą Nowicką", "influencerka kulinarna", "#f3cfb0", "#d8b25a", "#c0392b"
    ),
    "kalinowski": Persona(
        "prof. Andrzej Kalinowski",
        "prof. Andrzeja Kalinowskiego",
        "prof. Andrzejem Kalinowskim",
        "kardiolog",
        "#e6b590",
        "#cfcfcf",
        "#f4f4f4",
    ),
    "wrona": Persona(
        "Adam Wrona", "Adama Wrony", "Adamem Wroną", "dziennikarz ekonomiczny", "#dcae88", "#5a3d2b", "#34495e"
    ),
    "zielinska": Persona(
        "dr Barbara Zielińska",
        "dr Barbary Zielińskiej",
        "dr Barbarą Zielińską",
        "dietetyczka",
        "#f0c39e",
        "#7b3f00",
        "#ffffff",
    ),
    "malinowska": Persona(
        "Katarzyna Malinowska",
        "Katarzyny Malinowskiej",
        "Katarzyną Malinowską",
        "aktorka",
        "#f2c4a0",
        "#2c1b10",
        "#8e44ad",
    ),
    "sadowski": Persona(
        "Jan Sadowski",
        "Jana Sadowskiego",
        "Janem Sadowskim",
        "były piłkarz reprezentacji",
        "#d8a07a",
        "#4a3020",
        "#c0392b",
    ),
}

# ---------------------------------------------------------------- narracje (szablony tekstów)

NARRATIVES: dict[str, dict[str, Any]] = {
    "wywiad": {
        "title": "fałszywy wywiad z osobą publiczną → platforma inwestycyjna",
        "ads": [
            "{brand}: {person} ujawnia, jak zarabia {amount} tygodniowo. Ten wywiad zszokował Polaków!",
            "Skandal na antenie! {person} zdradził sposób na pasywny dochód. Banki chcą to zablokować.",
            "Wywiad z {person_ins}: „Każdy może zacząć od 1000 zł”. Przeczytaj, zanim usuną ten artykuł!",
            "{person} o platformie {product}: „W pierwszym tygodniu zarobiłem {amount}”.",
            "TYLKO DZIŚ: {person} pokazuje, jak {product} pomnaża oszczędności. Sprawdź szczegóły!",
            "Cała Polska mówi o wywiadzie z {person_ins}. Ten sposób na dochód działa naprawdę!",
        ],
        "headlines": [
            "{person} ujawnia sekret",
            "Skandal w studiu!",
            "{person}: tak zarabiam {amount}",
            "Ten wywiad zszokował Polaków",
        ],
        "paras": [
            "W rozmowie z redakcją {brand} {person}, {role}, opowiedział o platformie {product}, która według "
            "niego pozwala zarabiać nawet {amount} tygodniowo.",
            "Prowadzący próbował przerwać rozmowę, ale {person} zdążył pokazać wyniki swojego konta. Nagranie "
            "zniknęło z anteny kilka minut później.",
            "Aby zacząć, wystarczy wpłata 1000 zł. Algorytm platformy {product} sam wybiera najlepsze inwestycje.",
            "Redakcja przetestowała platformę: po 7 dniach saldo konta wzrosło z 1000 zł do 4 870 zł.",
        ],
        "quotes": [
            "„To najlepsza decyzja finansowa w moim życiu” — mówi {person}.",
            "„Nie potrzebujesz wiedzy, system robi wszystko za ciebie” — przekonuje {person}.",
        ],
        "ctas": ["Zarejestruj się teraz", "Sprawdź platformę", "Czytaj dalej"],
        "amount": (3000, 9000, 500),
    },
    "krypto": {
        "title": "inwestycje w kryptowaluty (giełda / platforma)",
        "ads": [
            "{person} inwestuje w kryptowaluty przez {product}. Zacznij od 250 € i odbieraj zyski codziennie! 🚀",
            "Bitcoin znów bije rekordy! {product} – giełda, którą poleca {person}. Bonus 100% na start.",
            "Nowa giełda {product}: automatyczne zyski z BTC i ETH. Dołącz do {count} Polaków!",
            "Odbierz 0,05 BTC na start w {product}. Oferta ważna tylko 24 godziny.",
            "Kryptowaluty dla każdego: {product} wypłaca nawet {amount} miesięcznie. Napisz na Telegram {telegram}.",
            "{person}: „Bitcoin to przyszłość, a {product} to najprostszy sposób, by na nim zarobić”.",
        ],
        "headlines": [
            "{product}: zyski z kryptowalut",
            "Bitcoin rośnie — dołącz",
            "{person} poleca {product}",
            "Bonus 100% na start",
        ],
        "paras": [
            "{product} to nowoczesna giełda kryptowalut, z której korzysta już {count} inwestorów z Polski.",
            "Wpłać minimum 250 € i odbieraj zyski z handlu bitcoinem i ethereum każdego dnia.",
            "Wsparcie klienta 24/7 na Telegramie: {telegram}. Wypłaty realizujemy w ciągu 24 godzin.",
            "Kryptowaluty zyskały w tym roku ponad 140%. Nie przegap kolejnej fali wzrostów.",
        ],
        "quotes": ["„Kryptowaluty zmieniły moje życie” — {person}.", "„Zacząłem od 250 euro” — {person}."],
        "ctas": ["Otwórz konto", "Kup bitcoina", "Odbierz bonus"],
        "amount": (8000, 30000, 1000),
    },
    "akcje": {
        "title": "akcje spółki skarbu państwa dla każdego",
        "ads": [
            "Polacy mogą kupić akcje {brand} już od 1000 zł. Gwarantowany zysk {amount} miesięcznie!",
            "Rządowy program: każdy obywatel może inwestować w {brand}. Rejestracja tylko do końca tygodnia.",
            "{person}: „Inwestycja w {brand} to bezpieczna przyszłość Twojej rodziny”. Zarejestruj się!",
            "Akcje {brand} dla każdego Polaka — nawet {amount} pasywnego dochodu. Liczba miejsc ograniczona.",
            "{product}: zainwestuj w polską energetykę i odbieraj dywidendę co miesiąc.",
        ],
        "headlines": [
            "Akcje {brand} dla każdego",
            "Zysk {amount} miesięcznie",
            "Inwestuj w polską energię",
            "Rejestracja do piątku",
        ],
        "paras": [
            "{brand} uruchamia {product}. Każdy pełnoletni obywatel może kupić pakiet akcji od 1000 zł.",
            "Według {person_gen}, {role}, to najbezpieczniejsza inwestycja dekady.",
            "Dywidenda wypłacana jest co miesiąc bezpośrednio na konto inwestora.",
            "Liczba miejsc w programie jest ograniczona — decyduje kolejność zgłoszeń.",
        ],
        "quotes": [
            "„To szansa dla każdej polskiej rodziny” — {person}.",
            "„Energetyka to fundament gospodarki” — podkreśla {person}.",
        ],
        "ctas": ["Kup akcje", "Zarejestruj się", "Sprawdź ofertę"],
        "amount": (2000, 7000, 500),
    },
    "doplata": {
        "title": "rządowa dopłata / świadczenie (wyłudzenie danych karty)",
        "ads": [
            "Odbierz dopłatę {amount} do rachunków za prąd! Wniosek online zajmuje 2 minuty.",
            "{brand}: przysługuje Ci zwrot {amount}. Sprawdź uprawnienia i podaj dane karty do wypłaty.",
            "Ostatnie dni! Rządowa dopłata {amount} dla każdego gospodarstwa domowego. Złóż wniosek.",
            "{person} ogłasza: {product} – {amount} dla emerytów i rodzin. Sprawdź, czy się kwalifikujesz.",
            "Nowe świadczenie {product}. Wypłata nawet {amount} w 48 godzin.",
        ],
        "headlines": ["Dopłata {amount} do rachunków", "{product}", "Złóż wniosek online", "Zwrot {amount}"],
        "paras": [
            "{product} to jednorazowe świadczenie dla gospodarstw domowych. Wypłata następuje na kartę płatniczą.",
            "Aby otrzymać {amount}, wypełnij formularz i potwierdź tożsamość danymi karty.",
            "{person}, {role}: „Pomoc trafi do każdego, kto złoży wniosek w terminie”.",
            "Wnioski przyjmowane są tylko do końca miesiąca.",
        ],
        "quotes": ["„Pomoc trafi do każdego” — {person}.", "„Wypłaty ruszają w tym tygodniu” — {person}."],
        "ctas": ["Złóż wniosek", "Odbierz dopłatę", "Sprawdź uprawnienia"],
        "amount": (600, 2500, 100),
    },
    "bot_ai": {
        "title": "inwestycje w kryptowaluty (bot AI do tradingu)",
        "ads": [
            "Sztuczna inteligencja handluje kryptowalutami za Ciebie! {product} zarabia nawet {amount} dziennie.",
            "{person}: „Mój bot AI zarobił więcej niż moja firma”. Sprawdź {product}.",
            "Algorytm AI wykrywa okazje na rynku bitcoina 24/7. {product} — skuteczność 94%!",
            "Zarejestruj się w {product} i pozwól algorytmowi pracować. Minimalna wpłata 1000 zł.",
            "Bot {product} podwoił oszczędności {count} inwestorów. Zacznij od 1000 zł, wypłata w 24h.",
        ],
        "headlines": [
            "Bot AI zarabia za Ciebie",
            "{product}: skuteczność 94%",
            "{amount} dziennie z AI",
            "Algorytm pracuje 24/7",
        ],
        "paras": [
            "{product} analizuje tysiące notowań kryptowalut na sekundę i sam zawiera transakcje.",
            "{person}, {role}, twierdzi, że bot zarobił dla niego {amount} w jeden dzień.",
            "Wystarczy 1000 zł wpłaty. Zyski wypłacisz w ciągu 24 godzin.",
            "Kontakt z doradcą przez Telegram: {telegram}.",
        ],
        "quotes": ["„Algorytm nie śpi i nie popełnia błędów” — {person}.", "„To przyszłość inwestowania” — {person}."],
        "ctas": ["Uruchom bota", "Zacznij teraz", "Zarejestruj się"],
        "amount": (1500, 6000, 250),
    },
    "sklep": {
        "title": "fałszywy sklep internetowy (wyprzedaż -80%)",
        "ads": [
            "Likwidacja magazynu! {product} za {price_new} zł zamiast {price_old} zł. Zostało tylko {stock} sztuk!",
            "{person} poleca: {product} -{discount}%! Darmowa dostawa i płatność przy odbiorze.",
            "WYPRZEDAŻ -{discount}% 🔥 {product} — ostatnie sztuki w sklepie {brand}.",
            "Zamów {product} dziś, a drugi otrzymasz gratis. Promocja kończy się o północy!",
        ],
        "headlines": ["-{discount}% na wszystko", "Likwidacja magazynu!", "Tylko {price_new} zł", "Ostatnie sztuki!"],
        "paras": [
            "{product} — oryginalny produkt z gwarancją 24 miesiące. Cena regularna {price_old} zł.",
            "Promocja obowiązuje tylko dziś. Zostało {stock} sztuk w magazynie.",
            "Płatność kartą lub przelewem natychmiastowym. Wysyłka w 24 godziny.",
            "{person}, {role}: „Używam go codziennie w swojej kuchni”.",
        ],
        "quotes": ["„Najlepszy zakup tego roku” — {person}.", "„Polecam wszystkim” — {person}."],
        "ctas": ["Kup teraz", "Zamów z rabatem", "Dodaj do koszyka"],
        "amount": (100, 200, 10),
    },
    "paczka": {
        "title": "dopłata do przesyłki (podszycie pod firmę kurierską)",
        "ads": [
            "{brand}: Twoja przesyłka została wstrzymana. Dopłać {small} zł, aby odebrać paczkę.",
            "Paczka nr {tracking} czeka na dopłatę {small} zł. Zapłać w ciągu 24h, inaczej wróci do nadawcy.",
            "Nie udało się doręczyć przesyłki — brak opłaty {small} zł. Uzupełnij dane adresowe.",
            "UWAGA: przesyłka {tracking} zatrzymana w sortowni. Opłać {small} zł online.",
        ],
        "headlines": ["Przesyłka wstrzymana", "Dopłać {small} zł", "Paczka {tracking}", "Odbierz paczkę"],
        "paras": [
            "Przesyłka {tracking} została zatrzymana w sortowni z powodu braku opłaty.",
            "Aby odebrać paczkę, uiść opłatę {small} zł kartą płatniczą w ciągu 24 godzin.",
            "Po tym czasie przesyłka zostanie zwrócona do nadawcy.",
            "{brand} — doręczamy w całej Polsce.",
        ],
        "quotes": ["Status: oczekuje na płatność.", "Status: zatrzymana w sortowni."],
        "ctas": ["Zapłać teraz", "Opłać przesyłkę", "Odbierz paczkę"],
        "amount": (100, 200, 10),
    },
    "lek": {
        "title": "cudowny lek / suplement polecany przez „lekarza”",
        "ads": [
            "{person}: „Ten naturalny preparat obniża ciśnienie w 7 dni”. Lekarze są w szoku!",
            "Koncerny farmaceutyczne ukrywają ten sposób! {product} — {person} ujawnia prawdę.",
            "Nadciśnienie znika bez tabletek? {person} poleca {product}. Pierwsze opakowanie za 0 zł.",
            "Wywiad z {person_ins}: jak {product} pomógł tysiącom pacjentów.",
            "{product} – naturalna formuła na serce. Zamów dziś z rabatem {discount}%.",
        ],
        "headlines": [
            "Lekarze w szoku!",
            "{product} na nadciśnienie",
            "{person} ujawnia prawdę",
            "Zdrowe serce w 7 dni",
        ],
        "paras": [
            "{person}, {role} z 30-letnim doświadczeniem, od lat bada naturalne metody leczenia nadciśnienia.",
            "{product} zawiera wyciąg z głogu i czosnku, który według badań obniża ciśnienie w 7 dni.",
            "Koncerny farmaceutyczne nie chcą, aby ta informacja trafiła do pacjentów.",
            "Zamów dziś, a pierwsze opakowanie otrzymasz za 0 zł. Infolinia: {phone}.",
        ],
        "quotes": ["„Moi pacjenci odstawili leki” — {person}.", "„To przełom w kardiologii” — {person}."],
        "ctas": ["Zamów teraz", "Odbierz rabat", "Sprawdź skład"],
        "amount": (100, 200, 10),
    },
    # ---- narracje tylko dla szumu (pojedyncze materiały)
    "kurs": {
        "title": "kurs gotowania online (reklama nieoszukańcza z tą samą osobą)",
        "ads": ["{person} zaprasza na kurs gotowania online. Pierwsza lekcja za darmo!"],
        "headlines": ["Gotuj z {person_ins}"],
        "paras": ["Kurs {product} prowadzi {person}, {role}."],
        "quotes": ["„Gotowanie to radość” — {person}."],
        "ctas": ["Zapisz się"],
        "amount": (100, 200, 10),
    },
    "loteria": {
        "title": "fałszywa loteria / nagroda",
        "ads": ["Gratulacje! Zostałeś wylosowany. Odbierz nagrodę od {brand} — zapłać tylko {small} zł za wysyłkę."],
        "headlines": ["Wygrałeś nagrodę!"],
        "paras": ["Nagroda {product} czeka na odbiór."],
        "quotes": ["Losowanie zakończone."],
        "ctas": ["Odbierz nagrodę"],
        "amount": (100, 200, 10),
    },
    "praca": {
        "title": "praca zdalna za polubienia",
        "ads": ["Praca zdalna! Zarabiaj {amount} dziennie, polubiając posty. Napisz na WhatsApp po instrukcję."],
        "headlines": ["Praca zdalna {amount} dziennie"],
        "paras": ["{brand} zatrudnia od zaraz."],
        "quotes": ["„Zarabiam z telefonu” — Ania z Łodzi."],
        "ctas": ["Aplikuj"],
        "amount": (200, 500, 50),
    },
    "pozyczka": {
        "title": "pożyczka bez BIK",
        "ads": ["Pożyczka bez BIK do {amount} w 15 minut. Bez zaświadczeń, decyzja od ręki — {brand}."],
        "headlines": ["Pożyczka bez BIK"],
        "paras": ["{brand}: pierwsza pożyczka za 0 zł."],
        "quotes": ["Decyzja w 15 minut."],
        "ctas": ["Weź pożyczkę"],
        "amount": (5000, 50000, 5000),
    },
    "olx": {
        "title": "fałszywa płatność w serwisie ogłoszeniowym",
        "ads": ["Kupujący opłacił Twój przedmiot. Potwierdź sprzedaż i odbierz płatność w serwisie {brand}."],
        "headlines": ["Płatność czeka na Ciebie"],
        "paras": ["Aby odebrać środki, podaj dane karty."],
        "quotes": ["Transakcja oczekuje."],
        "ctas": ["Odbierz płatność"],
        "amount": (100, 900, 50),
    },
    "fotowoltaika": {
        "title": "fotowoltaika z dotacją 100%",
        "ads": ["Fotowoltaika z dotacją 100%! {brand} — zostało tylko {stock} miejsc w programie."],
        "headlines": ["Fotowoltaika za 0 zł"],
        "paras": ["Program {product} obejmuje montaż i panele."],
        "quotes": ["„Rachunki spadły do zera”."],
        "ctas": ["Sprawdź dotację"],
        "amount": (100, 200, 10),
    },
    "paczka_n": {
        "title": "dopłata do przesyłki (inny przewoźnik)",
        "ads": ["Twoja paczka nie mogła zostać dostarczona. Potwierdź adres i opłać {small} zł w serwisie {brand}."],
        "headlines": ["Paczka czeka na odbiór"],
        "paras": ["Przesyłka czeka w punkcie odbioru {brand}."],
        "quotes": ["Status: oczekuje na potwierdzenie adresu."],
        "ctas": ["Potwierdź adres"],
        "amount": (100, 200, 10),
    },
    "krypto_n": {
        "title": "inwestycje w kryptowaluty (inny operator)",
        "ads": ["{person} radzi: kup bitcoina, zanim będzie za późno. Platforma {product} właśnie rusza w Polsce."],
        "headlines": ["{product} wchodzi do Polski"],
        "paras": ["{product} to nowa aplikacja do kupowania bitcoina."],
        "quotes": ["„Kupiłem pierwszego bitcoina za grosze” — {person}."],
        "ctas": ["Pobierz aplikację"],
        "amount": (1000, 5000, 500),
    },
    "sklep_n": {
        "title": "fałszywy sklep (inny produkt)",
        "ads": ["Wielka wyprzedaż w {brand}: {product} taniej o {discount}%. Dostawa gratis do końca tygodnia!"],
        "headlines": ["Taniej o {discount}%"],
        "paras": ["{product} w najniższej cenie w historii."],
        "quotes": ["„Dotarła w 2 dni” — Marta."],
        "ctas": ["Zamów"],
        "amount": (100, 200, 10),
    },
    "lek_n": {
        "title": "suplement odchudzający polecany przez „dietetyczkę”",
        "ads": ["{person}: „Schudłam 12 kg w miesiąc dzięki {product}”. Sprawdź naturalną metodę."],
        "headlines": ["Schudnij 12 kg w miesiąc"],
        "paras": ["{product} przyspiesza metabolizm."],
        "quotes": ["„Bez diety i ćwiczeń” — {person}."],
        "ctas": ["Zamów próbkę"],
        "amount": (100, 200, 10),
    },
    "angielski": {
        "title": "kurs języka angielskiego (reklama zwykła)",
        "ads": ["Kurs angielskiego online w {brand}. Pierwsza lekcja gratis, zapisy trwają."],
        "headlines": ["Angielski online"],
        "paras": ["Zajęcia z lektorem w małych grupach."],
        "quotes": ["„Polecam każdemu” — Kasia."],
        "ctas": ["Zapisz się"],
        "amount": (100, 200, 10),
    },
    "ankieta": {
        "title": "ankieta z voucherem",
        "ads": ["Wypełnij krótką ankietę i odbierz voucher {amount} do sklepów {brand}!"],
        "headlines": ["Voucher {amount} za ankietę"],
        "paras": ["Ankieta zajmuje 1 minutę."],
        "quotes": ["Zostało 12 voucherów."],
        "ctas": ["Wypełnij ankietę"],
        "amount": (500, 1500, 500),
    },
}

WORD_SYNONYMS: dict[str, list[str]] = {
    "zarabia": ["zarabia", "zgarnia", "inkasuje"],
    "zarobiłem": ["zarobiłem", "zgarnąłem"],
    "tygodniowo": ["tygodniowo", "co tydzień", "w tydzień"],
    "dziennie": ["dziennie", "każdego dnia", "na dzień"],
    "miesięcznie": ["miesięcznie", "co miesiąc"],
    "Sprawdź": ["Sprawdź", "Zobacz", "Kliknij i sprawdź"],
    "zszokował": ["zszokował", "zaskoczył", "poruszył"],
    "Polaków": ["Polaków", "widzów", "całą Polskę"],
    "Zarejestruj": ["Zarejestruj", "Zapisz"],
    "Dopłać": ["Dopłać", "Opłać", "Uiść"],
    "naturalny": ["naturalny", "ziołowy"],
    "ostatnie": ["ostatnie", "ostatnie już"],
    "Wypłata": ["Wypłata", "Przelew"],
}
EMOJI = ["🔥", "💰", "👉", "⚠️", "✅", "📈"]

# ---------------------------------------------------------------- kampanie bazowe


@dataclass
class CampaignSpec:
    slug: str
    narrative: str
    persona: str | None
    brand: str
    product: str
    layout: str  # układ kreacji
    page: str  # szkielet strony docelowej
    palette: tuple[str, str, str, str]  # tło, akcent, tekst, wyróżnienie
    cdn: bool  # za Cloudflare (wspólne ASN/IP/NS z innymi kampaniami)
    ns: str  # klucz serwerów nazw
    asn: str
    registrar: str
    n_domains: int
    redirect: str  # direct | cloak | shortener
    trackers: tuple[str, ...]  # ga, pixel, gtm, pixel_rot (inny piksel na każdej domenie)
    contact: tuple[str, ...]  # phone, telegram, btc, trx, eth
    san_cert: bool  # jeden certyfikat na wszystkie domeny
    favicon: str  # own | default
    platforms: tuple[str, ...]
    start: datetime
    notes: str = ""


ASN_CF = "AS13335 CLOUDFLARENET"
REGISTRARS = [
    "Example Domains Ltd.",
    "Przykładowy Rejestrator Sp. z o.o.",
    "TestNames Inc.",
    "Fikcyjne Domeny S.A.",
    "Demo Registrar GmbH",
]
NAMESERVERS = {
    "cf_a": ["anna.ns.cloudflare.example", "bob.ns.cloudflare.example"],
    "cf_b": ["kate.ns.cloudflare.example", "rick.ns.cloudflare.example"],
    "popular": ["ns1.popularny-dns.example", "ns2.popularny-dns.example"],
}
SHORTENER = "krotko.example"
PRIVACY_DOMAIN = "polityka-prywatnosci.example"
CF_IPS = [f"198.51.100.{i}" for i in range(10, 16)]


def _int32(s: str) -> int:
    return int.from_bytes(hashlib.sha1(s.encode()).digest()[:4], "big", signed=True)


DEFAULT_FAVICON = _int32("domyslna-favicona-hostingu")

BASE_CAMPAIGNS: list[CampaignSpec] = [
    CampaignSpec(
        "wywiad-zawadzki",
        "wywiad",
        "zawadzki",
        "Info24 Wiadomości",
        "Zyskomat Pro",
        "news",
        "article",
        ("#f4f6fa", "#1d4e9e", "#14213d", "#d62828"),
        True,
        "cf_a",
        ASN_CF,
        REGISTRARS[0],
        3,
        "cloak",
        ("ga", "pixel"),
        ("phone",),
        False,
        "own",
        ("facebook", "instagram"),
        datetime(2026, 9, 5),
        "ta sama osoba co kampania 02; wspólny Cloudflare/NS z 06",
    ),
    CampaignSpec(
        "krypto-zawadzki",
        "krypto",
        "zawadzki",
        "Biznes Express 24",
        "BitNova Exchange",
        "crypto",
        "crypto",
        ("#0b1020", "#19c37d", "#ffffff", "#f7b500"),
        False,
        "own",
        "AS64500 PRZYKLAD-HOSTING",
        REGISTRARS[1],
        4,
        "direct",
        ("pixel",),
        ("telegram", "btc", "trx"),
        True,
        "own",
        ("facebook", "youtube"),
        datetime(2026, 8, 12),
        "ta sama osoba co 01 (twardy negatyw); z 05 wspólna wyłącznie narracja",
    ),
    CampaignSpec(
        "akcje-krajenergia",
        "akcje",
        "wierzbicka",
        "KrajEnergia S.A.",
        "Program Akcjonariatu Obywatelskiego",
        "news2",
        "portal",
        ("#eef2f7", "#0b3d91", "#102a43", "#ff8c00"),
        True,
        "cf_b",
        ASN_CF,
        REGISTRARS[0],
        3,
        "cloak",
        ("ga",),
        ("phone",),
        False,
        "own",
        ("facebook", "google"),
        datetime(2026, 9, 22),
        "wspólny Cloudflare/NS/rejestrator z 08",
    ),
    CampaignSpec(
        "doplata-bon-energetyczny",
        "doplata",
        "domanski",
        "Fundusz Wsparcia Obywateli",
        "Bon Energetyczny 2026",
        "gov",
        "gov_form",
        ("#ffffff", "#dc143c", "#1a1a1a", "#003399"),
        False,
        "popular",
        "AS64502 BULLETPROOF-EXAMPLE",
        REGISTRARS[2],
        2,
        "shortener",
        (),
        (),
        False,
        "own",
        ("facebook",),
        datetime(2026, 10, 14),
        "ASN, NS, rejestrator i skracacz wspólne z 07",
    ),
    CampaignSpec(
        "bot-ai-neuroinwest",
        "bot_ai",
        "brzezinski",
        "TechNews Polska",
        "NeuroInwest AI",
        "bot",
        "bot",
        ("#1b0f3b", "#7b2ff7", "#ffffff", "#00e5ff"),
        False,
        "own",
        "AS64501 VPS-OMEGA-EXAMPLE",
        REGISTRARS[3],
        3,
        "direct",
        ("ga", "pixel_rot"),
        ("telegram", "eth"),
        True,
        "own",
        ("youtube", "tiktok", "facebook"),
        datetime(2026, 10, 1),
        "z 02 wspólna wyłącznie narracja",
    ),
    CampaignSpec(
        "sklep-kuchmistrz",
        "sklep",
        "nowicka",
        "Domowe Okazje",
        "robot kuchenny KuchMistrz 5000",
        "shop",
        "shop",
        ("#fffaf2", "#ff6a00", "#222222", "#e63946"),
        True,
        "cf_a",
        ASN_CF,
        REGISTRARS[1],
        3,
        "direct",
        ("ga", "pixel_rot"),
        ("phone",),
        False,
        "default",
        ("instagram", "facebook"),
        datetime(2026, 10, 20),
        "domyślna favicona hostingu (wspólna z 07 i szumem)",
    ),
    CampaignSpec(
        "paczka-doplata",
        "paczka",
        None,
        "Kurier Paczka24",
        "dopłata do przesyłki",
        "parcel",
        "parcel",
        ("#ffd400", "#5b3a1a", "#1a1a1a", "#d00000"),
        False,
        "popular",
        "AS64502 BULLETPROOF-EXAMPLE",
        REGISTRARS[2],
        4,
        "shortener",
        (),
        (),
        False,
        "default",
        ("facebook", "google"),
        datetime(2026, 11, 2),
        "bez osoby; wspólna infrastruktura z 04",
    ),
    CampaignSpec(
        "lek-kalinowski",
        "lek",
        "kalinowski",
        "Portal Zdrowie Dziś",
        "Kordelin Forte",
        "health",
        "health",
        ("#effaf3", "#2e8b57", "#123524", "#c1121f"),
        True,
        "cf_b",
        ASN_CF,
        REGISTRARS[0],
        2,
        "cloak",
        ("pixel", "gtm"),
        ("phone",),
        False,
        "own",
        ("facebook", "google"),
        datetime(2026, 10, 8),
        "wspólny Cloudflare/NS/rejestrator z 03",
    ),
]

# (narracja, osoba, typ układu/strony) dla materiałów-szumu
NOISE_PLAN = [  # układ kreacji inny niż kampanii z tą samą narracją (podobna narracja, nie ta sama grafika)
    ("kurs", "zawadzki", "news2"),
    ("paczka_n", None, "news"),
    ("krypto_n", "wrona", "crypto"),
    ("loteria", None, "shop"),
    ("praca", None, "bot"),
    ("pozyczka", None, "gov"),
    ("sklep_n", "malinowska", "news2"),
    ("lek_n", "zielinska", "health"),
    ("olx", None, "gov"),
    ("fotowoltaika", None, "health"),
    ("angielski", None, "news"),
    ("ankieta", "sadowski", "news2"),
]
PAGE_TYPES = ["article", "portal", "crypto", "gov_form", "bot", "shop", "parcel", "health"]
NOISE_BRANDS = [
    "Kulinarna Akademia",
    "PaczkoPost Express",
    "Finanse Dziś",
    "Wielka Loteria Konsumencka",
    "Praca Zdalna Plus",
    "Szybka Kasa 24",
    "Mega Okazje",
    "Natura Zdrowie",
    "Ogłoszenia Lokalne",
    "SłońceDom",
    "Lingua Online",
    "Bonus Market",
]
NOISE_PRODUCTS = [
    "Kurs Kuchni Domowej",
    "przesyłka kurierska",
    "CoinMax 24",
    "smartfon nagroda",
    "Polub i Zarabiaj",
    "pożyczka ekspresowa",
    "hulajnoga elektryczna Volta X",
    "SlimNatur",
    "płatność za ogłoszenie",
    "Program Słoneczny Dom",
    "Angielski w 3 miesiące",
    "voucher zakupowy",
]
HIJACKED_PAGES = [
    "Kwiaciarnia Pod Lipą",
    "Auto Serwis Marek",
    "Salon Fryzjerski Ania",
    "Pizzeria Bella",
    "Fundacja Zielony Dom",
    "Studio Fitness Forma",
    "Szkoła Tańca Rytm",
    "Piekarnia Złoty Kłos",
    "Hotel Pod Dębem",
    "Sklep Rowerowy Szprycha",
    "Gabinet Weterynaryjny Azor",
    "Kancelaria Lex",
    "Przedszkole Słoneczko",
    "Księgarnia Strona",
    "Warsztat Stolarski Dąb",
    "Cukiernia Wisienka",
]
DOMAIN_WORDS = {
    "wywiad": ["wywiad", "fakty", "info", "news", "raport", "dzis"],
    "krypto": ["bit", "coin", "krypto", "gielda", "trade", "nova"],
    "akcje": ["akcje", "energia", "inwestuj", "dywidenda", "kapital", "program"],
    "doplata": ["doplata", "wsparcie", "bon", "swiadczenie", "wniosek", "zwrot"],
    "bot_ai": ["ai", "bot", "neuro", "algo", "smart", "invest"],
    "sklep": ["okazje", "sklep", "promo", "wyprzedaz", "outlet", "magazyn"],
    "paczka": ["paczka", "kurier", "przesylka", "tracking", "dostawa", "sortownia"],
    "lek": ["zdrowie", "serce", "natura", "kardio", "apteka", "vita"],
}
GENERIC_WORDS = ["pl", "24", "online", "plus", "pro", "teraz", "polska", "dzis", "net", "go", "web", "hub"]
CLOAK_WORDS = ["go", "trk", "link", "click", "redir", "news", "flash", "jump", "route", "hop"]
CRM_WORDS = ["crm", "leads", "panel", "callcenter", "formy", "kontakt", "rejestracja", "api"]


# ---------------------------------------------------------------- pomocnicze: losowe identyfikatory

_B32 = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _bech32_polymod(values: list[int]) -> int:
    gen = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for v in values:
        top = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ v
        for i in range(5):
            chk ^= gen[i] if (top >> i) & 1 else 0
    return chk


def _btc_address(rng: random.Random) -> str:
    """Adres bech32 (P2WPKH) z poprawną sumą kontrolną — fikcyjny, losowy program."""
    prog = rng.randbytes(20)
    acc, bits, data = 0, 0, [0]
    for b in prog:
        acc = (acc << 8) | b
        bits += 8
        while bits >= 5:
            bits -= 5
            data.append((acc >> bits) & 31)
    if bits:
        data.append((acc << (5 - bits)) & 31)
    hrp = "bc"
    exp = [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]
    pm = _bech32_polymod(exp + data + [0] * 6) ^ 1
    checksum = [(pm >> 5 * (5 - i)) & 31 for i in range(6)]
    return hrp + "1" + "".join(_B32[d] for d in data + checksum)


def _trx_address(rng: random.Random) -> str:
    """Adres TRON (USDT-TRC20) — base58check z prefiksem 0x41."""
    payload = b"\x41" + rng.randbytes(20)
    raw = payload + hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    n = int.from_bytes(raw, "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = _B58[r] + out
    return out


def _eth_address(rng: random.Random) -> str:
    return "0x" + rng.randbytes(20).hex()


def _phone(rng: random.Random) -> str:
    return f"+48 5{rng.randint(0, 9)}{rng.randint(0, 9)} {rng.randint(100, 999)} {rng.randint(100, 999)}"


def _phone_variant(p: str, rng: random.Random) -> str:
    digits = p.replace("+48", "").replace(" ", "")
    return rng.choice([p, "+48" + digits, f"{digits[:3]}-{digits[3:6]}-{digits[6:]}", f"(+48) {digits}"])


def _ga(rng: random.Random) -> str:
    return "G-" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(10))


def _gtm(rng: random.Random) -> str:
    return "GTM-" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(7))


def _pixel(rng: random.Random) -> str:
    return str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([14, 15])))


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _fold(s: str) -> str:
    s = s.replace("ł", "l").replace("Ł", "L")
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def _slugify(s: str) -> str:
    out = "".join(ch if ch.isalnum() else "-" for ch in _fold(s).lower())
    return "-".join(p for p in out.split("-") if p)


# ---------------------------------------------------------------- kampania „skonkretyzowana”


@dataclass
class _Camp:
    idx: int
    label: str
    spec: CampaignSpec
    persona: Persona | None
    domains: list[str]
    cloak: str | None
    brand_domain: str
    crm: str
    ips: dict[str, list[str]]
    nameservers: list[str]
    created: dict[str, datetime]
    tls: dict[str, str]
    favicon: int
    ga: str | None
    gtm: str | None
    pixels: dict[str, str]
    phone: str | None
    telegram: str | None
    wallets: list[str]
    advertisers: list[str]
    templates: list[str]
    headlines: list[str]
    cls: str  # prefiks klas CSS (szkielet DOM kampanii)
    wrappers: int  # dodatkowe zagnieżdżenia w szkielecie
    geo: tuple[int, int]  # przesunięcie elementów kreacji (różni kampanie o tym samym układzie)
    is_noise: bool = False


class _World:
    """Wspólny stan generatora: unikalne domeny i adresy IP."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.domains: set[str] = {SHORTENER, PRIVACY_DOMAIN}
        self.next_ip = {"203.0.113": 20, "192.0.2": 20}
        self.used_templates: dict[str, list[str]] = {}
        self.hijacked = list(HIJACKED_PAGES)
        rng.shuffle(self.hijacked)

    def domain(self, words: list[str], tld: str | None = None) -> str:
        rng = self.rng
        for _ in range(200):
            w1, w2 = rng.sample(words, 2) if len(words) > 1 else (words[0], rng.choice(GENERIC_WORDS))
            if rng.random() < 0.4:
                w2 = rng.choice(GENERIC_WORDS)
            name = f"{w1}-{w2}" + (str(rng.randint(2, 99)) if rng.random() < 0.35 else "")
            d = f"{name}.{tld or rng.choice(['test', 'example'])}"
            if d not in self.domains:
                self.domains.add(d)
                return d
        raise RuntimeError("brak wolnych nazw domen")

    def brand_domain(self, brand: str) -> str:
        """Domena „podszywana” pod markę: info24wiadomosci.example (unikalna w zbiorze)."""
        base = _slugify(brand).replace("-", "")[:24] or "marka"
        d, k = f"{base}.example", 2
        while d in self.domains:
            d, k = f"{base}{k}.example", k + 1
        self.domains.add(d)
        return d

    def ip(self, net: str) -> str:
        n = self.next_ip[net]
        self.next_ip[net] = n + self.rng.randint(1, 7)
        return f"{net}.{n % 250 + 2}"

    def pages(self, k: int) -> list[str]:
        return [
            self.hijacked.pop() if self.hijacked else f"Strona Lokalna {self.rng.randint(10, 99)}" for _ in range(k)
        ]


def _make_campaign(
    idx: int, spec: CampaignSpec, world: _World, *, label: str | None = None, is_noise: bool = False
) -> _Camp:
    rng = world.rng
    words = DOMAIN_WORDS.get(spec.narrative, ["oferta", "promo", "info", "zysk", "start", "nowy"])
    domains = [world.domain(words) for _ in range(spec.n_domains)]
    cloak = world.domain(CLOAK_WORDS, "test") if spec.redirect == "cloak" else None
    brand_domain = world.brand_domain(spec.brand)
    crm = world.domain(CRM_WORDS, "test")
    if spec.cdn:
        ips = {d: sorted(rng.sample(CF_IPS, 2)) for d in domains}
        ns = NAMESERVERS[spec.ns]
    else:
        net = "192.0.2" if is_noise else "203.0.113"
        server = [world.ip(net) for _ in range(rng.choice([1, 2]))]
        ips = {d: list(server) for d in domains}
        if spec.ns in NAMESERVERS:
            ns = NAMESERVERS[spec.ns]
        else:
            base = world.domain(["dns", "serwer", "hosting", "vps", "omega", "alfa", "delta"], "test")
            ns = [f"ns1.{base}", f"ns2.{base}"]
    created: dict[str, datetime] = {}
    reg_day = spec.start - timedelta(days=rng.randint(5, 25))
    for d in domains:  # domeny rejestrowane hurtowo: w odstępie 0–3 dni
        created[d] = reg_day + timedelta(days=rng.randint(0, 3), hours=rng.randint(0, 23), minutes=rng.randint(0, 59))
    if spec.san_cert:
        fp = _sha256(f"san|{spec.slug}|{rng.random()}")
        tls = dict.fromkeys(domains, fp)
    else:
        tls = {d: _sha256(f"tls|{d}|{rng.random()}") for d in domains}
    favicon = DEFAULT_FAVICON if spec.favicon == "default" else _int32(f"favicon|{spec.slug}|{rng.random()}")
    pixel = _pixel(rng)
    pixels = (
        {d: (_pixel(rng) if "pixel_rot" in spec.trackers else pixel) for d in domains}
        if {"pixel", "pixel_rot"} & set(spec.trackers)
        else {}
    )
    persona = PERSONAS.get(spec.persona) if spec.persona else None
    all_tpl = NARRATIVES[spec.narrative]["ads"]
    k = min(len(all_tpl), 3 if len(all_tpl) >= 5 else len(all_tpl))
    used = world.used_templates.setdefault(spec.narrative, [])
    fresh = [t for t in all_tpl if t not in used]
    templates = (
        rng.sample(fresh, k)
        if len(fresh) >= k
        else fresh + rng.sample([t for t in all_tpl if t in used], k - len(fresh))
    )
    used.extend(t for t in templates if t not in used)
    headlines = list(NARRATIVES[spec.narrative]["headlines"])
    tg_handle = (
        "@"
        + rng.choice(["wsparcie", "support", "doradca", "inwestor", "admin"])
        + "_"
        + _slugify(spec.product).replace("-", "")[:12]
        + str(rng.randint(1, 99))
    )
    return _Camp(
        idx=idx,
        label=label or f"kampania-{idx:02d}-{spec.slug}",
        spec=spec,
        persona=persona,
        domains=domains,
        cloak=cloak,
        brand_domain=brand_domain,
        crm=crm,
        ips=ips,
        nameservers=list(ns),
        created=created,
        tls=tls,
        favicon=favicon,
        ga=_ga(rng) if "ga" in spec.trackers else None,
        gtm=_gtm(rng) if "gtm" in spec.trackers else None,
        pixels=pixels,
        phone=_phone(rng) if "phone" in spec.contact else None,
        telegram=tg_handle if "telegram" in spec.contact else None,
        wallets=[
            w
            for w in (
                _btc_address(rng) if "btc" in spec.contact else None,
                _trx_address(rng) if "trx" in spec.contact else None,
                _eth_address(rng) if "eth" in spec.contact else None,
            )
            if w
        ],
        advertisers=[spec.brand, *world.pages(2)],
        templates=templates,
        headlines=headlines,
        cls=rng.choice(["art", "nw", "pg", "lp", "cnt", "box", "ui", "wr"]) + str(rng.randint(1, 99)),
        wrappers=rng.randint(0, 2),
        geo=(0, 0) if not is_noise else (rng.randint(-40, 40), rng.randint(-25, 25)),
        is_noise=is_noise,
    )


def _random_palette(rng: random.Random) -> tuple[str, str, str, str]:
    """Losowa, ale czytelna paleta: tło, akcent i wyróżnienie pod biały tekst, tekst kontrastowy do tła."""

    def color(lo: int, hi: int) -> tuple[int, int, int]:
        while True:
            c = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            if lo <= lum <= hi:
                return c

    bg = color(150, 250) if rng.random() < 0.7 else color(10, 70)
    lum_bg = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    ink = (20, 20, 20) if lum_bg > 128 else (245, 245, 245)
    return tuple("#{:02x}{:02x}{:02x}".format(*c) for c in (bg, color(20, 100), ink, color(40, 110)))  # type: ignore[return-value]


def _extra_spec(i: int, rng: random.Random) -> CampaignSpec:
    """Kampanie ponad 8 bazowych: losowe połączenie narracji, osoby i infrastruktury."""
    base = BASE_CAMPAIGNS[i % len(BASE_CAMPAIGNS)]
    persona = rng.choice([p for p in PERSONAS if p != base.persona] + [None])
    brand = f"{rng.choice(['Nowe', 'Super', 'Pewne', 'Polskie', 'Twoje'])} {rng.choice(['Fakty', 'Okazje', 'Finanse', 'Zdrowie', 'Wiadomości'])} {rng.randint(2, 99)}"
    cdn = rng.random() < 0.5
    return CampaignSpec(
        slug=f"{base.narrative}-{i + 1:02d}",
        narrative=base.narrative,
        persona=persona,
        brand=brand,
        product=f"{rng.choice(['Max', 'Pro', 'Smart', 'Vita', 'Turbo'])}{rng.choice(['Zysk', 'Trade', 'Kasa', 'Forte', 'Plus'])} {rng.randint(2, 9)}",
        layout=base.layout,
        page=base.page,
        palette=_random_palette(rng),
        cdn=cdn,
        ns=rng.choice(["cf_a", "cf_b"]) if cdn else rng.choice(["popular", "own"]),
        asn=ASN_CF if cdn else f"AS{rng.randint(64503, 64510)} EXAMPLE-NET-{rng.randint(1, 9)}",
        registrar=rng.choice(REGISTRARS),
        n_domains=rng.randint(2, 4),
        redirect=rng.choice(["direct", "cloak", "shortener"]),
        trackers=tuple(t for t in ("ga", "pixel") if rng.random() < 0.6),
        contact=("phone",) if rng.random() < 0.5 else (),
        san_cert=rng.random() < 0.3,
        favicon="own",
        platforms=("facebook", "instagram"),
        start=datetime(2026, rng.randint(7, 11), rng.randint(1, 20)),
        notes="kampania dodatkowa (narracja wspólna z kampanią bazową)",
    )


def _noise_spec(k: int, rng: random.Random) -> CampaignSpec:
    narrative, persona, layout = NOISE_PLAN[k % len(NOISE_PLAN)]
    cdn = rng.random() < 0.3
    return CampaignSpec(
        slug=f"szum-{k + 1:02d}",
        narrative=narrative,
        persona=persona,
        brand=NOISE_BRANDS[k % len(NOISE_BRANDS)] + ("" if k < len(NOISE_BRANDS) else f" {k}"),
        product=NOISE_PRODUCTS[k % len(NOISE_PRODUCTS)],
        layout=layout,
        page=PAGE_TYPES[(3 * k + 1) % len(PAGE_TYPES)],  # równomiernie: ten sam szkielet najwyżej 2× na 12 szumów
        palette=_random_palette(rng),
        cdn=cdn,
        ns=rng.choice(["cf_a", "cf_b"]) if cdn else rng.choice(["popular", "own", "own"]),
        asn=ASN_CF if cdn else f"AS{rng.randint(64503, 64510)} EXAMPLE-NET-{rng.randint(1, 9)}",
        registrar=rng.choice(REGISTRARS),
        n_domains=1,
        redirect=rng.choice(["direct", "direct", "shortener"]),
        trackers=tuple(t for t in ("ga", "pixel") if rng.random() < 0.4),
        contact=("phone",) if rng.random() < 0.3 else (),
        san_cert=False,
        favicon="default" if rng.random() < 0.3 else "own",
        platforms=("facebook", "instagram", "google", "tiktok"),
        start=datetime(2026, rng.randint(6, 11), rng.randint(1, 25)),
    )


# ---------------------------------------------------------------- warianty tekstu


def _money(n: int, rng: random.Random) -> str:
    grouped = f"{n:,}".replace(",", " ")
    return rng.choice([f"{n} zł", f"{grouped} zł", f"{n} PLN", f"{grouped} złotych"])


def _slots(c: _Camp, rng: random.Random) -> dict[str, str]:
    nar = NARRATIVES[c.spec.narrative]
    lo, hi, step = nar["amount"]
    p = c.persona
    price_old = rng.choice([999, 1199, 1299, 1499])
    discount = rng.choice([70, 75, 80, 85, 88])
    return {
        "person": p.nom if p else c.spec.brand,
        "person_gen": p.gen if p else c.spec.brand,
        "person_ins": p.ins if p else c.spec.brand,
        "role": p.role if p else "rzecznik firmy",
        "brand": c.spec.brand,
        "product": c.spec.product,
        "amount": _money(rng.randrange(lo, hi + 1, step), rng),
        "count": f"{rng.randint(8, 40)} {rng.randint(100, 999)}",
        "telegram": c.telegram or "",
        "phone": c.phone or "",
        "small": f"{rng.randint(1, 2)},{rng.choice(['49', '79', '99'])}",
        "tracking": f"PX{rng.randint(100000000, 999999999)}PL",
        "price_old": str(price_old),
        "price_new": str(round(price_old * (100 - discount) / 100)),
        "discount": str(discount),
        "stock": str(rng.randint(7, 49)),
    }


def _fill(tpl: str, slots: dict[str, str]) -> str:
    class _Safe(dict):
        def __missing__(self, key: str) -> str:
            return ""

    return " ".join(tpl.format_map(_Safe(slots)).split())


def _typo(text: str, rng: random.Random) -> str:
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) >= 6 and w.isalpha()]
    if not idx:
        return text
    i = rng.choice(idx)
    w = words[i]
    kind = rng.randrange(4)
    j = rng.randint(1, len(w) - 3)
    if kind == 0:
        w = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    elif kind == 1:
        w = w[:j] + w[j + 1 :]
    elif kind == 2:
        w = _fold(w)
    else:
        w = w[:j] + w[j] + w[j:]
    words[i] = w
    return " ".join(words)


def _vary(text: str, rng: random.Random, protect: list[str]) -> str:
    for word, alts in WORD_SYNONYMS.items():
        if word in text and rng.random() < 0.5:
            text = text.replace(word, rng.choice(alts), 1)
    if rng.random() < 0.35:
        text = _typo(text, rng)
    if rng.random() < 0.08:  # reklama bez polskich znaków
        folded = _fold(text)
        for p in protect:  # identyfikatory (telefon, portfel) muszą zostać nietknięte
            folded = folded.replace(_fold(p), p)
        text = folded
    if rng.random() < 0.3:
        text = f"{rng.choice(EMOJI)} {text}" if rng.random() < 0.5 else f"{text} {rng.choice(EMOJI)}"
    return text


# ---------------------------------------------------------------- kreacje graficzne

_FONT_FILES = {
    False: [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
    ],
    True: [
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    ],
}


@lru_cache(maxsize=64)
def _font(size: int, bold: bool = False) -> tuple[Any, bool]:
    """(czcionka, czy ma polskie znaki). Bez czcionki systemowej → wbudowana w Pillow (tekst bez ogonków)."""
    for f in _FONT_FILES[bold]:
        try:
            return ImageFont.truetype(f, size), True
        except OSError:
            continue
    return ImageFont.load_default(size=size), False


def _t(s: str, size: int, bold: bool = False) -> tuple[str, Any]:
    font, pl = _font(size, bold)
    return (s if pl else _fold(s)), font


def _rgb(hexs: str, d: tuple[int, int, int] = (0, 0, 0)) -> tuple[int, int, int]:
    h = hexs.lstrip("#")
    return tuple(max(0, min(255, int(h[i : i + 2], 16) + d[k])) for k, i in enumerate((0, 2, 4)))  # type: ignore[return-value]


def _text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    size: int,
    fill: Any,
    max_w: int,
    bold: bool = True,
    max_lines: int = 4,
) -> int:
    s, font = _t(text, size, bold)
    lines, cur = [], ""
    for word in s.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    x, y = xy
    for line in lines[:max_lines]:
        draw.text((x, y), line, font=font, fill=fill)
        y += int(size * 1.22)
    return y


def _portrait(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], p: Persona | None, bg: Any) -> None:
    """Sylwetka osoby (bez oczu i ust): rysunek twarzy byłby wykrywany przez YuNet, a SFace uznawałby różne
    „osoby” za tę samą twarz — fałszywe przesłanki. Kolory skóry/włosów/stroju zależą od osoby."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    draw.rectangle(box, fill=bg)
    skin = _rgb(p.skin) if p else (200, 200, 200)
    hair = _rgb(p.hair) if p else (120, 120, 120)
    suit = _rgb(p.suit) if p else (90, 90, 90)
    draw.ellipse((x0 + w * 0.08, y0 + h * 0.66, x1 - w * 0.08, y1 + h * 0.4), fill=suit)
    draw.rectangle((x0 + w * 0.42, y0 + h * 0.52, x0 + w * 0.58, y0 + h * 0.7), fill=skin)
    draw.ellipse((x0 + w * 0.3, y0 + h * 0.16, x0 + w * 0.7, y0 + h * 0.6), fill=skin)
    draw.chord((x0 + w * 0.28, y0 + h * 0.1, x0 + w * 0.72, y0 + h * 0.46), 180, 360, fill=hair)


def _button(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill: Any, ink: Any = "white"
) -> None:
    draw.rounded_rectangle(box, radius=10, fill=fill)
    s, font = _t(text, 20, True)
    tw = draw.textlength(s, font=font)
    draw.text(((box[0] + box[2] - tw) / 2, box[1] + (box[3] - box[1] - 24) / 2), s, font=font, fill=ink)


def _creative(c: _Camp, headline: str, extra: str, cta: str, dcol: tuple[int, int, int]) -> Image.Image:
    """Kreacja reklamy: układ i kolory kampanii, nagłówek wariantu, sylwetka osoby (bez rysów twarzy)."""
    bg, acc, ink, hi = (_rgb(x, dcol) for x in c.spec.palette)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    gx, gy = c.geo
    lay = c.spec.layout
    brand = c.spec.brand
    p = c.persona
    if lay == "news":
        d.rectangle((0, 0, W, 56), fill=acc)
        _text_block(d, (16, 12), brand, 26, "white", 400, max_lines=1)
        d.rounded_rectangle((W - 130, 12, W - 16, 44), radius=6, fill=hi)
        _text_block(d, (W - 118, 15), "PILNE", 22, "white", 110, max_lines=1)
        _portrait(d, (24 + gx, 76 + gy, 224 + gx, 316 + gy), p, _rgb("#d9dde5", dcol))
        _text_block(d, (250 + gx, 84 + gy), headline, 28, ink, W - 270 - gx)
        _button(d, (250 + gx, 322, 540 + gx, 372), cta, hi)
    elif lay == "news2":
        d.rectangle((0, H - 70, W, H), fill=acc)
        d.rectangle((20, 20, 90, 90), fill=hi)
        _text_block(d, (100, 34), brand, 24, ink, 300, max_lines=1)
        _portrait(d, (380 + gx, 40 + gy, 580 + gx, 320 + gy), p, _rgb("#c9d6e8", dcol))
        _text_block(d, (24, 110 + gy), headline, 30, ink, 340 + gx)
        _text_block(d, (24, H - 52), extra, 22, "white", 540, max_lines=1)
    elif lay == "crypto":
        pts = [(0, 330), (80, 300), (150, 315), (230, 250), (300, 270), (380, 190), (450, 210), (W, 120)]
        d.line([(x, y + gy) for x, y in pts], fill=acc, width=6)
        _text_block(d, (24, 24), c.spec.product, 34, "white", 360, max_lines=2)
        _text_block(d, (24, 120 + gy), headline, 24, hi, 330)
        cx, cy = 470 + gx, 160 + gy
        d.ellipse((cx - 100, cy - 100, cx + 100, cy + 100), fill=acc)
        _portrait(d, (cx - 80, cy - 80, cx + 80, cy + 80), p, acc)
        _button(d, (24, 330, 260, 380), cta, hi, "black")
    elif lay == "gov":
        d.rectangle((0, 0, W, 18), fill="white")
        d.rectangle((0, 18, W, 36), fill=acc)
        _text_block(d, (20, 50), brand, 24, ink, 560, max_lines=1)
        d.rounded_rectangle((20 + gx, 96, 300 + gx, 196), radius=12, fill=hi)
        _text_block(d, (36 + gx, 120), extra, 36, "white", 250, max_lines=1)
        _text_block(d, (20, 214), headline, 26, ink, 360, max_lines=3)
        _portrait(d, (430 + gx, 110 + gy, 580 + gx, 300 + gy), p, _rgb("#e6e6e6", dcol))
        _button(d, (20, 330, 300, 380), cta, acc)
    elif lay == "bot":
        for i in range(0, H, 8):
            d.line([(0, i), (W, i)], fill=_rgb(c.spec.palette[0], (i // 16, 0, i // 10)), width=8)
        d.rounded_rectangle((420 + gx, 40 + gy, 560 + gx, 180 + gy), radius=14, outline=hi, width=5)
        _text_block(d, (462 + gx, 88 + gy), "AI", 40, hi, 90, max_lines=1)
        for k in range(5):
            d.line([(436 + gx + k * 28, 180 + gy), (436 + gx + k * 28, 205 + gy)], fill=hi, width=4)
        _portrait(d, (24, 40, 164, 200), p, acc)
        _text_block(d, (180, 50), c.spec.product, 30, "white", 230, max_lines=2)
        _text_block(d, (24, 230), headline, 26, "white", 540, max_lines=2)
        _button(d, (24, 330, 300, 380), cta, acc)
    elif lay == "shop":
        _text_block(d, (20, 16), brand, 26, acc, 560, max_lines=1)
        d.rounded_rectangle((30 + gx, 70 + gy, 270 + gx, 300 + gy), radius=16, fill=_rgb("#dddddd", dcol))
        d.ellipse((70 + gx, 120 + gy, 230 + gx, 260 + gy), fill=acc)
        d.ellipse((380 + gx, 60, 540 + gx, 220), fill=hi)
        _text_block(d, (412 + gx, 112), extra, 40, "white", 140, max_lines=1)
        _text_block(d, (300, 240), headline, 26, ink, 280, max_lines=2)
        _button(d, (300, 330, 570, 380), cta, acc)
    elif lay == "parcel":
        d.rectangle((40 + gx, 90 + gy, 240 + gx, 260 + gy), fill=acc)
        d.line([(40 + gx, 150 + gy), (240 + gx, 150 + gy)], fill=_rgb("#c8a26b"), width=10)
        d.line([(140 + gx, 90 + gy), (140 + gx, 260 + gy)], fill=_rgb("#c8a26b"), width=10)
        _text_block(d, (20, 16), brand, 28, ink, 560, max_lines=1)
        _text_block(d, (270, 90), headline, 30, ink, 310, max_lines=3)
        _text_block(d, (270, 230), extra, 34, hi, 310, max_lines=1)
        _button(d, (270, 320, 560, 372), cta, hi)
    else:  # health
        _text_block(d, (20, 16), brand, 26, acc, 560, max_lines=1)
        _portrait(d, (20 + gx, 64 + gy, 200 + gx, 300 + gy), p, _rgb("#d8efe1", dcol))
        d.rounded_rectangle((470 + gx, 150, 560 + gx, 330), radius=18, fill=acc)
        d.rectangle((470 + gx, 210, 560 + gx, 260), fill="white")
        d.rounded_rectangle((220, 70, 440, 110), radius=8, fill=hi)
        _text_block(d, (232, 76), "Lekarze w szoku", 22, "white", 200, max_lines=1)
        _text_block(d, (220, 130), headline, 26, ink, 240, max_lines=4)
        _button(d, (220, 330, 450, 380), cta, hi)
    return img


def _degrade(img: Image.Image, rng: random.Random) -> tuple[Image.Image, bool]:
    """Wariant tej samej grafiki: przycięcie, skalowanie, szum, rekompresja JPEG, rzadko odbicie lustrzane."""
    noisy = False
    if rng.random() < 0.6:
        w, h = img.size
        box = (
            rng.randint(0, w // 33),
            rng.randint(0, h // 33),
            w - rng.randint(0, w // 33),
            h - rng.randint(0, h // 33),
        )
        img = img.crop(box)
    img = img.resize(
        rng.choice([(600, 400), (600, 400), (640, 427), (560, 373), (720, 480), (480, 320)]), Image.Resampling.BICUBIC
    )
    if rng.random() < 0.1:
        nprng = np.random.default_rng(rng.randrange(1 << 30))
        arr = np.asarray(img, dtype=np.float32) + nprng.normal(0, 3.0, (img.size[1], img.size[0], 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        noisy = True
    if rng.random() < 0.25:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(60, 85))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        noisy = True
    if rng.random() < 0.04:
        img = ImageOps.mirror(img)
    return img, noisy


def _png_bytes(img: Image.Image, noisy: bool) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=4 if noisy else 6)
    return buf.getvalue()


# ---------------------------------------------------------------- strony docelowe


def _trackers_html(ga: str | None, gtm: str | None, pixel: str | None) -> str:
    out = []
    if ga:
        out.append(
            f'<script async src="https://www.googletagmanager.com/gtag/js?id={ga}"></script>\n'
            "<script>window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}"
            f"gtag('js',new Date());gtag('config','{ga}');</script>"
        )
    if gtm:
        out.append(
            "<script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':new Date().getTime(),event:'gtm.js'});"
            "var f=d.getElementsByTagName(s)[0],j=d.createElement(s);j.async=true;"
            f"j.src='https://www.googletagmanager.com/gtm.js?id='+i;f.parentNode.insertBefore(j,f);}})(window,document,'script','dataLayer','{gtm}');</script>"
        )
    if pixel:
        out.append(
            "<script>!function(f,b,e,v,n,t,s){if(f.fbq)return;n=f.fbq=function(){n.callMethod?"
            "n.callMethod.apply(n,arguments):n.queue.push(arguments)};if(!f._fbq)f._fbq=n;n.push=n;n.loaded=!0;"
            "n.version='2.0';n.queue=[];t=b.createElement(e);t.async=!0;t.src=v;s=b.getElementsByTagName(e)[0];"
            "s.parentNode.insertBefore(t,s)}(window,document,'script','https://connect.facebook.net/en_US/fbevents.js');"
            f"fbq('init','{pixel}');fbq('track','PageView');</script>"
        )
    return "\n".join(out)


def _form(action: str, fields: list[tuple[str, str]], button: str, cls: str) -> str:
    inputs = "".join(f'<label>{escape(lbl)}<input type="text" name="{name}"></label>' for name, lbl in fields)
    return f'<form class="{cls}-form" action="{escape(action)}" method="post">{inputs}<button type="submit">{escape(button)}</button></form>'


def _page(c: _Camp, v: dict[str, Any]) -> str:
    """Strona docelowa wariantu — szkielet DOM wspólny dla kampanii, treść zmienna."""
    e = escape
    k = c.cls
    paras = "".join(f"<p>{e(x)}</p>" for x in v["paras"])
    comments = "".join(f'<div class="{k}-comment"><b>{e(n)}</b><span>{e(t)}</span></div>' for n, t in v["comments"])
    contact = []
    if c.phone:
        contact.append(f"Infolinia: {e(_phone_variant(c.phone, v['rng']))}")
    if c.telegram:
        contact.append(f'Telegram: <a href="https://t.me/{e(c.telegram.lstrip("@"))}">{e(c.telegram)}</a>')
    for w in c.wallets:
        contact.append(f"Wpłaty na adres portfela: <code>{e(w)}</code>")
    footer_links = (
        f'<a href="https://{PRIVACY_DOMAIN}/{_slugify(c.spec.brand)}">Polityka prywatności</a> · '
        f'<a href="https://{c.brand_domain}/">{e(c.spec.brand)}</a> · <a href="/regulamin">Regulamin</a>'
    )
    footer = f'<footer class="{k}-foot"><p>{" · ".join(contact)}</p><p>{footer_links}</p><p>© 2026 {e(c.spec.brand)}</p></footer>'
    form = _form(f"https://{c.crm}/api/lead", v["fields"], v["cta"], k)
    pg = c.spec.page
    hero_img = f'<img src="/img/{_slugify(v["headline"])[:40]}.jpg" alt="{e(v["person"])}">'
    if pg == "article":
        body = (
            f'<header class="{k}-top"><div class="{k}-logo">{e(c.spec.brand)}</div><nav><a href="/">Wiadomości</a>'
            f'<a href="/biznes">Biznes</a><a href="/kraj">Kraj</a><a href="/sport">Sport</a></nav></header>'
            f'<main class="{k}-main"><article class="{k}-article"><h1>{e(v["headline"])}</h1>'
            f'<div class="{k}-meta">Redakcja {e(c.spec.brand)} · {e(v["date"])}</div>{hero_img}'
            f'<p class="lead">{e(v["lead"])}</p><blockquote>{e(v["quote"])}</blockquote>{paras}'
            f'<a class="{k}-cta" href="#rejestracja">{e(v["cta"])}</a></article>'
            f'<section id="rejestracja"><h2>Zarejestruj się na platformie {e(c.spec.product)}</h2>{form}</section>'
            f'<section class="{k}-comments"><h3>Komentarze</h3>{comments}</section></main>{footer}'
        )
    elif pg == "portal":
        body = (
            f'<div class="{k}-bar">Serwis informacyjny · {e(v["date"])}</div>'
            f'<div class="{k}-wrap"><div class="{k}-crumbs"><a href="/">Start</a> › <a href="/gospodarka">Gospodarka</a></div>'
            f'<div class="{k}-cols"><aside class="{k}-side"><h4>Najczęściej czytane</h4><ul>'
            + "".join(f"<li><a href='/a/{i}'>{e(t)}</a></li>" for i, t in enumerate(v["related"]))
            + f'</ul><div class="{k}-box"><h4>{e(c.spec.product)}</h4>{form}</div></aside>'
            f'<section class="{k}-content"><h1>{e(v["headline"])}</h1>{hero_img}<p><strong>{e(v["lead"])}</strong></p>'
            f"{paras}<blockquote>{e(v['quote'])}</blockquote></section></div></div>{footer}"
        )
    elif pg == "crypto":
        body = (
            f'<section class="{k}-hero"><h1>{e(c.spec.product)}</h1><h2>{e(v["headline"])}</h2>'
            f'<div class="{k}-stats"><span>{e(v["count"])} użytkowników</span><span>Wypłaty 24h</span><span>Bonus 100%</span></div>'
            f'<a class="{k}-btn" href="#signup">{e(v["cta"])}</a></section>'
            f'<section class="{k}-features"><div class="card"><h3>Bezpieczeństwo</h3><p>{e(v["paras"][0])}</p></div>'
            f'<div class="card"><h3>Szybkie wypłaty</h3><p>{e(v["paras"][-1])}</p></div>'
            f'<div class="card"><h3>Ekspert poleca</h3><p>{e(v["quote"])}</p></div></section>'
            f'<section id="signup" class="{k}-signup">{form}</section>'
            f'<section class="{k}-reviews">{comments}</section>{footer}'
        )
    elif pg == "gov_form":
        body = (
            f'<header class="{k}-gov"><span class="{k}-flag"></span><b>{e(c.spec.brand)}</b><span>Serwis obywatelski</span></header>'
            f'<main><h1>{e(v["headline"])}</h1><p class="{k}-lead">{e(v["lead"])}</p><ol class="{k}-steps">'
            "<li>Wypełnij formularz</li><li>Potwierdź tożsamość kartą</li><li>Odbierz środki</li></ol>"
            f'<div class="{k}-panel">{form}</div><div class="{k}-info">{paras}</div></main>{footer}'
        )
    elif pg == "bot":
        body = (
            f'<div class="{k}-nav"><b>{e(c.spec.product)}</b><a href="#start">Start</a></div>'
            f'<section class="{k}-video"><div class="player">▶</div><h1>{e(v["headline"])}</h1><p>{e(v["lead"])}</p></section>'
            f'<section class="{k}-how"><h2>Jak to działa?</h2><div class="step">1. Rejestracja</div>'
            '<div class="step">2. Wpłata 1000 zł</div><div class="step">3. Bot AI zarabia</div></section>'
            f'<table class="{k}-calc"><tr><th>Wpłata</th><th>Zysk po 30 dniach</th></tr>'
            "<tr><td>1000 zł</td><td>12 400 zł</td></tr><tr><td>5000 zł</td><td>61 800 zł</td></tr></table>"
            f'<section id="start" class="{k}-form-wrap"><h2>{e(v["cta"])}</h2>{form}</section>'
            f'<section class="{k}-testimonials">{comments}</section>{footer}'
        )
    elif pg == "shop":
        body = (
            f'<header class="{k}-shop"><b>{e(c.spec.brand)}</b><span>Koszyk (0)</span></header>'
            f'<main class="{k}-product"><div class="{k}-gallery">{hero_img}</div><div class="{k}-buy">'
            f'<h1>{e(c.spec.product)}</h1><p class="old">{e(v["price_old"])} zł</p><p class="new">{e(v["price_new"])} zł</p>'
            f'<p class="{k}-timer">Promocja kończy się za 02:14:37</p><h2>{e(v["headline"])}</h2>{form}</div></main>'
            f'<section class="{k}-desc">{paras}</section><section class="{k}-reviews"><h3>Opinie</h3>{comments}</section>{footer}'
        )
    elif pg == "parcel":
        body = (
            f'<header class="{k}-courier"><b>{e(c.spec.brand)}</b></header><main class="{k}-track">'
            f'<h1>{e(v["headline"])}</h1><div class="{k}-status"><p>Numer przesyłki: <b>{e(v["tracking"])}</b></p>'
            "<ul><li>Nadana</li><li>W sortowni</li><li><b>Wstrzymana — brak opłaty</b></li></ul></div>"
            f'<div class="{k}-pay"><p>Do zapłaty: {e(v["small"])} zł</p>{form}</div>{paras}</main>{footer}'
        )
    else:  # health
        body = (
            f'<header class="{k}-health"><b>{e(c.spec.brand)}</b><nav><a href="/">Zdrowie</a><a href="/serce">Serce</a></nav></header>'
            f'<article class="{k}-art"><h1>{e(v["headline"])}</h1><div class="{k}-doctor">{hero_img}'
            f"<p><b>{e(v['person'])}</b> — {e(v['role'])}</p></div><p>{e(v['lead'])}</p>{paras}"
            f'<blockquote>{e(v["quote"])}</blockquote><ul class="{k}-symptoms"><li>bóle głowy</li><li>zmęczenie</li>'
            f'<li>kołatanie serca</li></ul></article><aside class="{k}-order"><h2>{e(c.spec.product)}</h2>'
            f'<p>Rabat {e(v["discount"])}%</p>{form}</aside><section class="{k}-comments">{comments}</section>{footer}'
        )
    if c.is_noise:  # szum: własny wariant szkieletu (inny układ sekcji), żeby nie udawał tej samej kampanii
        r = v["rng"]
        extra = "".join(
            f'<section class="{k}-x{i}"><h3>{e(t)}</h3><p>{e(t)}</p></section>'
            for i, t in enumerate(r.sample(NOISE_SECTIONS, r.randint(1, 4)))
        )
        body = f'<div class="{k}-banner">{e(c.spec.brand)}</div>{extra}{body}' if r.random() < 0.5 else body + extra
    for _ in range(c.wrappers):
        body = f'<div class="{k}-container">{body}</div>'
    return (
        f'<!DOCTYPE html>\n<html lang="pl">\n<head>\n<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>{e(v["title"])}</title>\n'
        f'<meta name="description" content="{e(v["lead"][:150])}">\n<link rel="icon" href="{v["favicon"]}">\n'
        f'<link rel="stylesheet" href="/static/{k}.css">\n{v["trackers"]}\n</head>\n<body class="{k}-body">\n{body}\n</body>\n</html>\n'
    )


COMMENT_NAMES = [
    "Anna K.",
    "Krzysztof W.",
    "Magda",
    "Paweł z Gdańska",
    "Ewelina",
    "Tadeusz",
    "Joanna M.",
    "Robert",
    "Grażyna",
    "Michał P.",
]
COMMENTS = [
    "Działa! Pierwsza wypłata po 3 dniach.",
    "Na początku nie wierzyłam, ale polecam.",
    "Szkoda, że tak późno się dowiedziałem.",
    "Mój mąż też zaczął, jesteśmy zadowoleni.",
    "Czy to legalne? Ktoś sprawdzał?",
    "Super sprawa, dziękuję redakcji!",
    "Zamówiłam, czekam na paczkę.",
    "Polecam wszystkim znajomym.",
]
FIELDS = {
    "article": [("imie", "Imię"), ("nazwisko", "Nazwisko"), ("email", "E-mail"), ("telefon", "Telefon")],
    "portal": [("name", "Imię i nazwisko"), ("phone", "Telefon"), ("email", "E-mail")],
    "crypto": [("first_name", "First name"), ("last_name", "Last name"), ("email", "Email"), ("phone", "Phone")],
    "gov_form": [
        ("imie", "Imię"),
        ("nazwisko", "Nazwisko"),
        ("pesel", "PESEL"),
        ("numer_karty", "Numer karty"),
        ("waznosc", "Data ważności"),
        ("cvv", "CVV"),
    ],
    "bot": [("fname", "Imię"), ("lname", "Nazwisko"), ("mail", "E-mail"), ("tel", "Telefon")],
    "shop": [("imie_nazwisko", "Imię i nazwisko"), ("telefon", "Telefon"), ("adres", "Adres"), ("miasto", "Miasto")],
    "parcel": [("card_number", "Numer karty"), ("exp", "MM/RR"), ("cvc", "CVC"), ("holder", "Posiadacz karty")],
    "health": [("imie", "Imię"), ("telefon", "Telefon")],
}
NOISE_FIELDS = [
    ("imie", "Imię"),
    ("name", "Imię i nazwisko"),
    ("email", "E-mail"),
    ("mail", "Adres e-mail"),
    ("telefon", "Telefon"),
    ("phone", "Numer telefonu"),
    ("miasto", "Miasto"),
    ("kod", "Kod pocztowy"),
    ("wiek", "Wiek"),
    ("uwagi", "Wiadomość"),
]
NOISE_SECTIONS = [
    "Często zadawane pytania",
    "O nas",
    "Jak to działa",
    "Bezpieczeństwo danych",
    "Opinie klientów",
    "Kontakt",
    "Nasze certyfikaty",
    "Dostawa i zwroty",
    "Partnerzy",
    "Newsletter",
]


# ---------------------------------------------------------------- rekordy


@dataclass
class _Item:
    camp: _Camp
    label: str
    text: str | None
    url: str | None
    infra: dict[str, Any] | None
    page: dict[str, Any] | None  # parametry strony (None = brak HTML)
    creatives: list[tuple[str, str, str, tuple[int, int, int], int]]  # (nagłówek, extra, cta, przesunięcie, ziarno)
    platform: str | None
    advertiser: str | None
    date: datetime | None
    date_fmt: int = 0
    dup_of: _Item | None = None
    id: str = ""
    image_files: list[str] = field(default_factory=list)
    html_file: str | None = None


def _infra_for(c: _Camp, domain: str, url: str, final: str, rng: random.Random, dead: bool) -> dict[str, Any]:
    chain = [url] if url == final else [url, final]
    inf: dict[str, Any] = {
        "redirect_chain": chain,
        "final_url": final,
        "http_status": 200,
        "ips": list(c.ips[domain]),
        "asns": [c.spec.asn],
        "nameservers": list(c.nameservers),
        "registrar": c.spec.registrar,
        "domain_created": c.created[domain].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tls_issuer": "Example Edge CA WE1" if c.spec.cdn else "Example CA R10",
        "tls_fingerprint": c.tls[domain],
        "tls_sans": sorted(c.domains) if c.spec.san_cert else [domain, f"www.{domain}"],
        "favicon_mmh3": c.favicon,
    }
    if dead:  # strona zdjęta: brak HTML, częściowe dane (RDAP bywa dostępny dłużej niż DNS)
        err, status = rng.choice(
            [
                ("DNS: NXDOMAIN — domena nie istnieje", None),
                ("HTTP 404 — strona usunięta", 404),
                ("przekroczono limit czasu połączenia", None),
                ("HTTP 451 — domena zablokowana", 451),
            ]
        )
        inf = {"redirect_chain": [url], "final_url": None, "http_status": status, "errors": [f"pobieranie: {err}"]}
        if status:
            inf.update({"ips": list(c.ips[domain]), "asns": [c.spec.asn]})
        if rng.random() < 0.5:
            inf.update(
                {"registrar": c.spec.registrar, "domain_created": c.created[domain].strftime("%Y-%m-%dT%H:%M:%SZ")}
            )
    return inf


def _variants(c: _Camp, n: int, rng: random.Random) -> list[_Item]:
    nar = NARRATIVES[c.spec.narrative]
    dates = sorted(
        c.spec.start + timedelta(days=rng.randint(0, 28), hours=rng.randint(6, 22), minutes=rng.randint(0, 59))
        for _ in range(n)
    )
    items: list[_Item] = []
    for i, when in enumerate(dates):
        slots = _slots(c, rng)
        protect = [x for x in (c.phone, c.telegram, *c.wallets) if x]
        text = _vary(_fill(rng.choice(c.templates), slots), rng, protect)
        domain = c.domains[min(len(c.domains) - 1, i * len(c.domains) // n)]  # rotacja domen w czasie
        path = (
            rng.choice(["artykul", "oferta", "pl", "start", "promo", "lp"])
            if c.is_noise
            else {
                "article": "artykul",
                "portal": "gospodarka",
                "crypto": "pl/rejestracja",
                "gov_form": "wniosek",
                "bot": "start",
                "shop": "produkt",
                "parcel": "sledzenie",
                "health": "zdrowie",
            }[c.spec.page]
        )
        final = f"https://{domain}/{path}/{_slugify(_fill(rng.choice(nar['headlines']), slots))[:40]}"
        if rng.random() < 0.4:
            final += f"?utm_source={rng.choice(['fb', 'ig', 'yt', 'gads'])}&utm_campaign={rng.randint(100, 999)}"
        if c.spec.redirect == "cloak" and c.cloak:
            url = f"https://{c.cloak}/{rng.choice(['r', 'go', 'c'])}/{rng.randbytes(4).hex()}"
        elif c.spec.redirect == "shortener":
            url = f"https://{SHORTENER}/{rng.randbytes(3).hex()}"
        else:
            url = final
        dead = rng.random() < 0.12
        infra = _infra_for(c, domain, url, final, rng, dead)
        headline = _fill(rng.choice(c.headlines), slots)
        extra = {
            "gov": slots["amount"],
            "shop": f"-{slots['discount']}%",
            "parcel": f"{slots['small']} zł",
            "news2": c.spec.product,
        }.get(c.spec.layout, slots["amount"])
        cta = rng.choice(nar["ctas"])
        dcol = (rng.randint(-8, 8), rng.randint(-8, 8), rng.randint(-8, 8))
        creatives = [(headline, extra, cta, dcol, rng.randrange(1 << 30))]
        if rng.random() < 0.06:  # karuzela: drugi obraz z tej samej kampanii
            creatives.append((_fill(rng.choice(c.headlines), slots), extra, cta, dcol, rng.randrange(1 << 30)))
        persona = c.persona
        page = None
        body_paras = nar["paras"][1:] or nar["paras"]  # pierwszy akapit to lead — bez powtórzeń
        if not dead:
            page = {
                "title": f"{headline} | {c.spec.brand}",
                "headline": headline,
                "lead": _fill(nar["paras"][0], slots),
                "paras": [_fill(x, slots) for x in rng.sample(body_paras, min(3, len(body_paras)))],
                "quote": _fill(rng.choice(nar["quotes"]), slots),
                "cta": cta,
                "comments": list(zip(rng.sample(COMMENT_NAMES, 5), rng.sample(COMMENTS, 5), strict=True))[
                    : rng.randint(3, 5)
                ],
                "related": [_fill(h, slots) for h in nar["headlines"][:3]],
                "person": persona.nom if persona else c.spec.brand,
                "role": persona.role if persona else "",
                "date": when.strftime("%d.%m.%Y"),
                "count": slots["count"],
                "small": slots["small"],
                "tracking": slots["tracking"],
                "price_old": slots["price_old"],
                "price_new": slots["price_new"],
                "discount": slots["discount"],
                "fields": rng.sample(NOISE_FIELDS, rng.randint(2, 4)) if c.is_noise else FIELDS[c.spec.page],
                "favicon": "/favicon.ico" if c.spec.favicon == "default" else f"/static/{c.cls}-icon.png",
                "trackers": _trackers_html(c.ga, c.gtm, c.pixels.get(domain)),
                "rng_seed": rng.randrange(1 << 30),
            }
        items.append(
            _Item(
                camp=c,
                label=c.label,
                text=text,
                url=url,
                infra=infra,
                page=page,
                creatives=creatives,
                platform=rng.choice(c.spec.platforms),
                advertiser=rng.choice(c.advertisers) if rng.random() > 0.12 else None,
                date=when if rng.random() > 0.06 else None,
                date_fmt=1 if rng.random() < 0.12 else 0,
            )
        )
    return items


def _make_incomplete(items: list[_Item], rng: random.Random) -> None:
    """~20% rekordów bez obrazu, adresu lub tekstu (część „zapisanych stron” bez adresu)."""
    for it in items:
        if rng.random() >= 0.2:
            continue
        what = rng.choice(["image", "url", "text"])
        if what == "image":
            it.creatives = []
        elif what == "text":
            it.text = None
        else:
            it.url = None
            it.infra = None
            if rng.random() < 0.5:
                it.page = None


def _duplicate(orig: _Item, rng: random.Random) -> _Item:
    c = orig.camp
    when = (orig.date or c.spec.start) + timedelta(days=rng.randint(1, 6), hours=rng.randint(0, 12))
    return _Item(
        camp=c,
        label=orig.label,
        text=orig.text,
        url=orig.url,
        infra=orig.infra,
        page=orig.page,
        creatives=orig.creatives,
        platform=rng.choice(c.spec.platforms),
        advertiser=rng.choice(c.advertisers),
        date=when,
        dup_of=orig,
    )


# ---------------------------------------------------------------- zapis


def _record(it: _Item) -> dict[str, Any]:
    image: Any = None
    if it.image_files:
        image = it.image_files[0] if len(it.image_files) == 1 else list(it.image_files)
    date = None
    if it.date:
        date = it.date.strftime("%d.%m.%Y %H:%M") if it.date_fmt else it.date.strftime("%Y-%m-%dT%H:%M:%S")
    return {
        "id": it.id,
        "url": it.url,
        "text": it.text,
        "image": image,
        "html": it.html_file,
        "platform": it.platform,
        "advertiser": it.advertiser,
        "date": date,
        "campaign": it.label,
        "infra": it.infra,
    }


def _readme(camps: list[_Camp], noise: list[_Camp], items: list[_Item], seed: int) -> str:
    n = len(items)
    no_img = sum(1 for it in items if not it.image_files)
    no_url = sum(1 for it in items if not it.url)
    no_text = sum(1 for it in items if not it.text)
    dead = sum(1 for it in items if it.infra and it.infra.get("errors"))
    dups = sum(1 for it in items if it.dup_of is not None)
    lines = [
        "# Syntetyczny zbiór reklam-oszustw (Ślad Fejka)",
        "",
        f"Wygenerowany przez `slad.ingest.synthetic.generate(seed={seed})` — deterministycznie, w całości fikcyjny.",
        "Służy do testów i demonstracji, dopóki nie ma danych NASK. Osoby, marki, produkty, telefony, portfele",
        "i identyfikatory trackerów są zmyślone; domeny wyłącznie `.test` / `.example`, adresy IP z bloków TEST-NET",
        "(192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24), ASN z puli dokumentacyjnej (poza celowo wspólnym",
        "`AS13335 CLOUDFLARENET`).",
        "",
        "## Pliki",
        "",
        "- `records.jsonl` — rekordy: `id, url, text, image, html, platform, advertiser, date, campaign, infra`",
        "  (`campaign` = etykieta wzorcowa, tylko do ewaluacji; `infra` = gotowe dane sieciowe zgodne z `InfraFeatures`,",
        "  dzięki czemu potok działa offline),",
        "- `images/` — kreacje PNG (układ i kolory kampanii, nagłówek wariantu, sylwetka osoby bez rysów twarzy; warianty",
        "  różnią się nagłówkiem, odcieniem, przycięciem, skalą, szumem i rekompresją JPEG),",
        "- `pages/` — strony docelowe (wspólny szkielet DOM kampanii, tytuł, gtag/GTM/Pixel, formularz leadowy,",
        "  favicona, linki wychodzące).",
        "",
        "## Statystyki",
        "",
        f"- rekordów: **{n}**, kampanii: **{len(camps)}**, materiałów-szumu (unikalne etykiety): **{len(noise)}**",
        f"- bez obrazu: {no_img}, bez adresu: {no_url}, bez tekstu: {no_text}, martwe strony: {dead}, "
        f"dokładne duplikaty: {dups}",
        "",
        "## Kampanie",
        "",
        "| etykieta | narracja | osoba | marka / medium | domeny | infrastruktura | uwagi |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in camps:
        size = sum(1 for it in items if it.label == c.label)
        infra = ("Cloudflare" if c.spec.cdn else c.spec.asn) + f"; {c.spec.registrar}"
        if c.spec.redirect != "direct":
            infra += "; przekierowanie " + (
                "przez skracacz" if c.spec.redirect == "shortener" else "przez domenę-pośrednika"
            )
        lines.append(
            f"| `{c.label}` ({size}) | {NARRATIVES[c.spec.narrative]['title']} | {c.persona.nom if c.persona else '—'} | "
            f"{c.spec.brand} | {', '.join(c.domains)} | {infra} | {c.spec.notes} |"
        )
    lines += [
        "",
        "## Trudne przypadki",
        "",
        "- **(a) ta sama osoba w różnych kampaniach** — kampanie 01 i 02 (oraz jeden materiał-szum) używają tej samej",
        "  fikcyjnej osoby; poza nazwiskiem nie łączy ich nic istotnego,",
        "- **(b) wspólna infrastruktura** — Cloudflare (ASN, adresy IP, pary serwerów nazw) dla 01/03/06/08,",
        "  popularny DNS, rejestrator, ASN i skracacz `krotko.example` dla 04/07, domyślna favicona hostingu,",
        "  wspólny generator polityki prywatności — każda z tych cech osobno nie powinna łączyć kampanii,",
        "- **(c) niekompletne dane** — ok. 20% rekordów bez obrazu / adresu / tekstu; martwe strony bez HTML,",
        "  z błędami pobierania i częściowymi danymi infrastruktury,",
        "- **(d) duplikaty** — ponownie opublikowane reklamy (ta sama grafika bajt w bajt i ten sam tekst),",
        "- **(e) szum** — pojedyncze materiały z etykietami `noise-01`, `noise-02`, … (inne oszustwa, ale też zwykłe",
        "  reklamy), część z tą samą narracją lub osobą co kampanie,",
        "- **(f) wspólna tylko narracja** — kampanie 02 (giełda kryptowalut) i 05 (bot AI) dzielą wyłącznie narrację",
        "  „inwestycje w kryptowaluty”.",
        "",
        "Wspólne w obrębie kampanii (sygnały, które POWINNY łączyć): identyfikatory GA4/GTM/Pixel, endpoint formularza,",
        "szkielet DOM strony, favicona, certyfikat TLS (SAN), telefon / Telegram / portfel, domena-pośrednik,",
        "serwer (IP) poza Cloudflare, daty rejestracji domen, układ graficzny kreacji, szablony tekstu.",
        "",
        "Ponowne wygenerowanie: `python -c \"from slad.ingest.synthetic import generate; generate('data/synthetic')\"`",
        "",
    ]
    return "\n".join(lines)


def generate(out_dir: Path | str, seed: int = 7, n_campaigns: int = 8, noise: int = 12) -> Path:
    """Generuje zbiór syntetyczny w ``out_dir`` i zwraca ścieżkę do ``records.jsonl``."""
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    world = _World(rng)

    specs = [BASE_CAMPAIGNS[i] if i < len(BASE_CAMPAIGNS) else _extra_spec(i, rng) for i in range(max(0, n_campaigns))]
    camps = [_make_campaign(i + 1, s, world) for i, s in enumerate(specs)]
    noise_camps = [
        _make_campaign(100 + k, _noise_spec(k, rng), world, label=f"noise-{k + 1:02d}", is_noise=True)
        for k in range(max(0, noise))
    ]

    items: list[_Item] = []
    for c in camps:
        vs = _variants(c, rng.randint(4, 9), rng)
        _make_incomplete(vs, rng)
        complete = [v for v in vs if v.creatives and v.text]
        for _ in range(rng.choice([0, 1, 1, 2]) if complete else 0):
            vs.append(_duplicate(rng.choice(complete), rng))
        items.extend(vs)
    for c in noise_camps:
        v = _variants(c, 1, rng)[0]
        _make_incomplete([v], rng)
        items.append(v)

    rng.shuffle(items)
    width = max(3, len(str(len(items))))
    for i, it in enumerate(items, 1):
        it.id = f"syn-{i:0{width}d}"

    # grafiki i strony: najpierw oryginały, potem duplikaty (kopiują bajty oryginału)
    rendered: dict[int, list[bytes]] = {}
    for it in sorted(items, key=lambda x: x.dup_of is not None):
        if it.dup_of is not None:
            blobs = rendered.get(id(it.dup_of), [])
            it.html_file = it.dup_of.html_file
        else:
            blobs = []
            for headline, extra, cta, dcol, img_seed in it.creatives:
                img, noisy = _degrade(_creative(it.camp, headline, extra, cta, dcol), random.Random(img_seed))
                blobs.append(_png_bytes(img, noisy))
            if it.page is not None:
                page = dict(it.page)
                page["rng"] = random.Random(page.pop("rng_seed"))
                it.html_file = f"pages/{it.id}.html"
                (out / it.html_file).write_text(_page(it.camp, page), encoding="utf-8")
            rendered[id(it)] = blobs
        for k, data in enumerate(blobs):
            name = f"images/{it.id}.png" if k == 0 else f"images/{it.id}-{k + 1}.png"
            (out / name).write_bytes(data)
            it.image_files.append(name)

    records = out / "records.jsonl"
    with records.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(_record(it), ensure_ascii=False) + "\n")
    (out / "README.md").write_text(_readme(camps, noise_camps, items, seed), encoding="utf-8")
    log.info(
        "Zbiór syntetyczny: %d rekordów, %d kampanii, %d szumu → %s", len(items), len(camps), len(noise_camps), out
    )
    return records
