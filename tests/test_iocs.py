"""Wskaźniki IOC z tekstu: wartości znormalizowane, kolejność, i — równie ważne — brak fałszywych trafień."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:  # bez instalacji pakietu / conftest.py
    sys.path.insert(0, str(ROOT))

from slad.features import iocs  # noqa: E402
from slad.features.iocs import IOC_KINDS, extract_iocs, fallback_registered_domain, iban_valid  # noqa: E402


@pytest.fixture(autouse=True)
def _no_infra(monkeypatch):
    """Testy nie zależą od modułu infrastruktury (pisanego równolegle): wersje uproszczone domen i URL."""
    monkeypatch.setattr(iocs, "_INFRA", {"registered_domain": None, "normalize_url": None})


def found(text: str) -> dict[str, list[str]]:
    return {k: v for k, v in extract_iocs(text).items() if v}


def test_all_keys_present_and_empty_input():
    for text in ("", "   ", None, "Zwykły tekst bez wskaźników."):
        out = extract_iocs(text)  # type: ignore[arg-type]
        assert list(out) == list(IOC_KINDS)
        assert all(v == [] for v in out.values())


@pytest.mark.parametrize(
    ("text", "phone"),
    [
        ("Zadzwoń: +48 600 123 456", "+48600123456"),
        ("tel. 0048 601-234-567", "+48601234567"),
        ("kom. 512345678", "+48512345678"),
        ("Infolinia (22) 123 45 67", "+48221234567"),
        ("+48 (22) 123 45 67", "+48221234567"),
        ("Tel:+48600123456", "+48600123456"),
        ("48 600 123 456", "+48600123456"),
        ("numer 221234567", "+48221234567"),
        ("UK desk +44 20 7946 0958.", "+442079460958"),
        ("+1 (555) 123-4567", "+15551234567"),
        ("czynne 24/7: 600 123 458 24/7", "+48600123458"),
        ("2026-11-21 600 123 456", "+48600123456"),
    ],
)
def test_phones_normalized_to_e164(text, phone):
    assert found(text).get("phone") == [phone]


@pytest.mark.parametrize(
    "text",
    [
        "Data 21.11.2026, termin 2026-11-21, godz. 12:30",
        "21-11-2026 123 osób kupiło",
        "Kwota 100 000 000 zł i 500 000 000 zł",
        "Zarobki 512 000 000 PLN",
        "Dołącz do 600 000 Polaków",
        "Cena 199,99 zł, zysk 300%",
        "NIP 123-456-32-18, PESEL 85010112345, REGON: 123456785",
        "Nr zamówienia: 600123456",
        "ID: 600123456",
        "Kod 00-950 Warszawa",
        "Pixel 123456789012345, reklama 120208123456789012",
        "+10 000 000 zł na start",
        "221234567 sztuk",
        "0.512345678 BTC",
    ],
)
def test_no_phone_false_positives(text):
    assert "phone" not in found(text)


def test_iban_digits_never_become_phone_numbers():
    out = found("Konto: 61 1090 1014 0000 0712 1981 2874, błędne: PL61 1090 1014 0000 0712 1981 2875")
    assert out == {"iban": ["PL61109010140000071219812874"]}


def test_emails_lowercased_deduplicated_and_refanged():
    out = found("Pisz: Kontakt@Scam-Example.PL lub kontakt@scam-example.pl albo biuro[at]inwestycje24.pl; logo@2x.png")
    assert out["email"] == ["kontakt@scam-example.pl", "biuro@inwestycje24.pl"]
    assert "domain" not in out  # domena z e-maila to nie adres strony


def test_urls_and_registered_domains():
    text = (
        "Rejestracja: https://orlen-invest.site/rejestracja?ref=fb&utm_source=fb&fbclid=IwAR0x. Zobacz "
        "www.inwestuj-teraz.pl/start), zarabiaj24.online! oraz hxxps://scam[.]com/x i http://192.168.1.1/admin, "
        "a także sub.panel.orlen-pl.com.pl/x i https://płatności-pko.pl/login"
    )
    out = found(text)
    assert out["url"] == [
        "https://orlen-invest.site/rejestracja?ref=fb",  # parametry śledzące usunięte
        "http://www.inwestuj-teraz.pl/start",
        "http://zarabiaj24.online/",
        "https://scam.com/x",
        "http://192.168.1.1/admin",
        "http://sub.panel.orlen-pl.com.pl/x",
        "https://xn--patnoci-pko-t5b34a.pl/login",  # IDN → punycode (jak w module infrastruktury)
    ]
    assert out["domain"] == [
        "orlen-invest.site",
        "inwestuj-teraz.pl",
        "zarabiaj24.online",
        "scam.com",
        "orlen-pl.com.pl",
        "xn--patnoci-pko-t5b34a.pl",
    ]


def test_same_result_with_infra_module(monkeypatch):
    """Z modułem infrastruktury (PSL, normalizacja URL) wynik ma ten sam kształt co w wersji uproszczonej."""
    pytest.importorskip("slad.features.infra.domains")
    text = (
        "https://orlen-invest.site/rejestracja?ref=fb&utm_source=fb zarabiaj24.online sub.scam-host.wordpress.com/post"
    )
    expected = found(text)
    monkeypatch.setattr(iocs, "_INFRA", {})
    assert found(text) == expected


def test_platform_domains_are_not_campaign_domains():
    out = found(
        "Sponsorowane · facebook.com · gov.pl · onet.pl — https://www.facebook.com/fanpage123 bit.ly/Abc123 "
        "sub.scam-host.wordpress.com/post"
    )
    assert out["url"] == [
        "https://www.facebook.com/fanpage123",
        "http://bit.ly/Abc123",
        "http://sub.scam-host.wordpress.com/post",
    ]
    assert out["domain"] == ["scam-host.wordpress.com"]  # hosting: liczy się konkretna strona


def test_hosting_domain_same_with_and_without_psl(monkeypatch):
    # moduł infra z PSL bez domen prywatnych zwróciłby „wordpress.com” — wynik ma być ten sam
    monkeypatch.setattr(iocs, "_INFRA", {"registered_domain": lambda host: "wordpress.com", "normalize_url": None})
    assert found("sub.scam-host.wordpress.com/post")["domain"] == ["scam-host.wordpress.com"]


def test_no_url_false_positives():
    assert found("Plik raport.pdf, wersja 2.5, np. tzw. koniec.Teraz, e-mail jan@firma.pl") == {
        "email": ["jan@firma.pl"]
    }


def test_fallback_registered_domain():
    assert fallback_registered_domain("a.b.example.co.uk") == "example.co.uk"
    assert fallback_registered_domain("WWW.Scam.COM.PL.") == "scam.com.pl"
    assert fallback_registered_domain("panel.oszustwo.pl") == "oszustwo.pl"
    assert fallback_registered_domain("x.github.io") == "x.github.io"
    assert fallback_registered_domain("10.0.0.1") == ""


def test_crypto_wallets():
    text = (
        "Wpłać BTC: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa lub 3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy, "
        "SegWit BC1QAR0SRRR7XFKVY5L643LYDNW9RE59GTZZWF5MDQ; ETH 0x52908400098527886E0F7030069857D2E4169EE7; "
        "USDT TRC20: TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"
    )
    out = found(text)
    assert out["crypto_btc"] == [
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy",
        "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq",
    ]
    assert out["crypto_eth"] == ["0x52908400098527886e0f7030069857d2e4169ee7"]
    assert out["crypto_trx"] == ["TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"]
    assert "phone" not in out


def test_crypto_checksum_and_context():
    # poprawna suma kontrolna → wskaźnik nawet bez kontekstu
    assert found("adres 1BoatSLRHtKNngkdXEeobR76b53LETtpyT")["crypto_btc"] == ["1BoatSLRHtKNngkdXEeobR76b53LETtpyT"]
    # zmyślony adres (np. zbiór syntetyczny) — tylko w kontekście kryptowalut
    fake = "1AbCdEfGhJkLmNpQrStUvWxYz23456789"
    assert found(f"Portfel bitcoin: {fake}")["crypto_btc"] == [fake]
    assert found(f"Kod referencyjny {fake} w regulaminie") == {}
    # hash transakcji (64 znaki hex) to nie adres ETH
    assert found("tx 0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060") == {}


def test_telegram_and_whatsapp():
    text = (
        "Kanał: https://t.me/InwestujZNami, grupa t.me/+AbCdEf12345, telegram.me/joinchat/XyZ98765. "
        "Napisz na Telegramie: @Doradca_PL. WhatsApp: wa.me/48600123459 albo "
        "https://api.whatsapp.com/send?phone=48600123460&text=hej, grupa chat.whatsapp.com/AbCdEfGhIjKlMn"
    )
    out = found(text)
    assert out["telegram"] == ["@inwestujznami", "t.me/+AbCdEf12345", "t.me/+XyZ98765", "@doradca_pl"]
    assert out["whatsapp"] == [
        "+48600123459",
        "+48600123460",
    ]  # tylko numery (moduł powiązań normalizuje je jak telefony)
    assert out["url"] == ["https://chat.whatsapp.com/AbCdEfGhIjKlMn"]  # zaproszenie do grupy: unikalny adres
    assert "domain" not in out and "phone" not in out


def test_at_handle_without_telegram_context_is_ignored():
    assert found("Obserwuj nas na Instagramie @inwestycje_pl") == {}


def test_iban():
    assert iban_valid("PL61 1090 1014 0000 0712 1981 2874")
    assert not iban_valid("PL61 1090 1014 0000 0712 1981 2875")
    out = found("IBAN DE89 3704 0044 0532 0130 00, GB29 NWBK 6016 1331 9268 19, PL61109010140000071219812874")
    assert out["iban"] == ["DE89370400440532013000", "GB29NWBK60161331926819", "PL61109010140000071219812874"]
    assert found("Konto: 12 3456 7890 1234 5678 9012 3456") == {}  # zła suma kontrolna


def test_order_of_appearance_and_dedup():
    out = found("tel. 600 123 457, potem +48 600 123 456 i znów 600-123-457")
    assert out["phone"] == ["+48600123457", "+48600123456"]


def test_never_raises_on_odd_input():
    weird = "\x00\ufeff" + "9" * 5000 + " http://[::1]/x https://a.b:99999/ " + "@" * 100 + " 1" * 3000
    out = extract_iocs(weird)
    assert list(out) == list(IOC_KINDS)
