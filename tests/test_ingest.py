"""Wczytywanie zbiorów: formaty, mapowanie kolumn PL/EN, katalogi, identyfikatory, odporność na złe dane."""

from __future__ import annotations

import base64
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.ingest import load_dataset, map_column, material_from_input, normalize_url, parse_date  # noqa: E402
from slad.models import Material, SourceKind  # noqa: E402
from slad.store import Workspace  # noqa: E402


def _png(path: Path, color: str = "red", size: tuple[int, int] = (40, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _png_bytes(color: str = "blue") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), color).save(buf, "PNG")
    return buf.getvalue()


def _by_id(ms: list[Material]) -> dict[str, Material]:
    return {m.id: m for m in ms}


# ---------------------------------------------------------------- mapowanie i normalizacja


@pytest.mark.parametrize(
    ("column", "field"),
    [
        ("Lp.", "id"),
        ("ID", "id"),
        ("Treść", "text"),
        ("Treść reklamy", "text"),
        ("Adres URL", "url"),
        ("Link do reklamy", "url"),
        ("landingUrl", "url"),
        ("Zrzut ekranu", "image"),
        ("url_obrazu", "image"),
        ("screenshot_url", "image"),
        ("Kampania", "label"),
        ("nazwa_kampanii", "label"),
        ("Data publikacji", "date"),
        ("Platforma", "platform"),
        ("Reklamodawca", "advertiser"),
        ("page_title", "text"),
        ("ad_creative_link_titles", "text"),
        ("html_file", "html"),
        ("infra", "infra"),
        ("Domena", "domain"),
        ("IP", "infra:ips"),
        ("adres_email", None),
        ("telefon", None),
        ("creative_id", None),
        ("kategoria", None),
    ],
)
def test_map_column(column: str, field: str | None) -> None:
    assert map_column(column)[0] == field


def test_normalize_url() -> None:
    assert normalize_url("hxxps://scam[.]pl/x") == "https://scam.pl/x"
    assert normalize_url("zysk.test/a?b=1") == "http://zysk.test/a?b=1"
    assert normalize_url("www.example.com") == "http://www.example.com"
    assert normalize_url("//cdn.example.com/a") == "https://cdn.example.com/a"
    for bad in ("ad1.png", "Zarabiaj teraz", "3.5", "ftp://x.pl/a", "https://x.pl/a b", "", None):
        assert normalize_url(bad) is None


def test_parse_date_formats() -> None:
    assert parse_date("2026-11-21") == datetime(2026, 11, 21)
    assert parse_date("2026-11-21T14:30:00Z") == datetime(2026, 11, 21, 14, 30)  # UTC, bez strefy
    assert parse_date("2026-11-21T15:30:00+01:00") == datetime(2026, 11, 21, 14, 30)
    assert parse_date("21.11.2026") == datetime(2026, 11, 21)
    assert parse_date("21.11.2026 14:30") == datetime(2026, 11, 21, 14, 30)
    assert parse_date("21 listopada 2026, 14:30") == datetime(2026, 11, 21, 14, 30)
    assert parse_date("1763740800") == datetime(2025, 11, 21, 16, 0)
    assert parse_date(1763740800000) == datetime(2025, 11, 21, 16, 0)
    assert parse_date(datetime(2026, 1, 2, 3, 4)) == datetime(2026, 1, 2, 3, 4)
    for bad in ("3", "K1", "", None, "brak", "2026"):
        assert parse_date(bad) is None


# ---------------------------------------------------------------- pliki rekordów


def test_csv_polish_headers_semicolon_cp1250(tmp_path: Path) -> None:
    _png(tmp_path / "images" / "ad1.png")
    rows = [
        "Lp.;Treść;Adres URL;Zrzut ekranu;Kampania;Data;Kategoria;Platforma",
        "1;Zarabiaj 5000 zł dziennie;zysk-teraz.test/start;ad1.png;K1;21.11.2026;inwestycje;FB",
        '2;"Tekst; ze średnikiem";hxxps://inna[.]test/x;brak.png;;2026-11-20 10:00;sklep;instagram',
    ]
    (tmp_path / "dane.csv").write_bytes("\n".join(rows).encode("cp1250"))
    ms = _by_id(load_dataset(tmp_path / "dane.csv"))
    m1, m2 = ms["1"], ms["2"]
    assert m1.text == "Zarabiaj 5000 zł dziennie"
    assert m1.url == "http://zysk-teraz.test/start"
    assert m1.image_paths == [str((tmp_path / "images" / "ad1.png").resolve())]
    assert m1.label == "K1"
    assert m1.observed_at == datetime(2026, 11, 21)
    assert m1.platform == "facebook"
    assert m1.meta["Kategoria"] == "inwestycje"
    assert m1.source == SourceKind.RECORD
    assert "K1" not in json.dumps(m1.meta)  # etykieta nie może wyciec do meta (cechy)
    assert m2.text == "Tekst; ze średnikiem"
    assert m2.url == "https://inna.test/x"
    assert m2.image_paths == [] and m2.label is None
    assert any("brak.png" in w for w in m2.meta["ingest_warnings"])


def test_json_shapes(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps([{"id": 1, "url": "https://a.test"}, {"id": 2, "text": "x"}]))
    (tmp_path / "b.json").write_text(json.dumps({"meta": {"v": 1}, "records": [{"tekst": "abc"}, {"link": "b.test"}]}))
    (tmp_path / "c.json").write_text(
        json.dumps({"r1": {"url": "https://c.test", "tekst": "t"}, "r2": {"url": "https://d.test", "tekst": "u"}})
    )
    a = load_dataset(tmp_path / "a.json")
    assert [m.id for m in a] == ["1", "2"] and a[0].source == SourceKind.URL and a[1].source == SourceKind.TEXT
    b = load_dataset(tmp_path / "b.json")
    assert [m.text for m in b] == ["abc", None] and b[1].url == "http://b.test"
    c = _by_id(load_dataset(tmp_path / "c.json"))
    assert set(c) == {"r1", "r2"} and c["r1"].url == "https://c.test"


def test_jsonl_bad_lines_infra_inline_html_and_data_uri(tmp_path: Path) -> None:
    data_uri = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()
    lines = [
        json.dumps(
            {
                "id": "r1",
                "landing_url": "https://lp.test/a",
                "ad_url": "https://www.facebook.example/ads/1",
                "ad_text": "Bitcoin",
                "title": "Nagłówek",
                "images": ["https://cdn.example.com/x.jpg", data_uri],
                "page_html": "<!DOCTYPE html><html><body><p>Hi</p></body></html>",
                "infra": {"ips": "192.0.2.1, 192.0.2.2", "asns": "AS13335 CLOUDFLARENET", "favicon_mmh3": "123"},
                "campaign": "K7",
                "extra_field": {"nested": True},
            }
        ),
        "{to nie jest json",
        json.dumps("https://tylko-adres.test/x"),
        "",
    ]
    (tmp_path / "records.jsonl").write_text("\n".join(lines), encoding="utf-8")
    ms = load_dataset(tmp_path / "records.jsonl")
    assert len(ms) == 2
    m = ms[0]
    assert m.url == "https://lp.test/a"  # strona docelowa ma pierwszeństwo przed adresem reklamy
    assert m.meta["urls"] == ["https://www.facebook.example/ads/1"]
    assert m.text == "Bitcoin\nNagłówek"
    assert m.meta["image_urls"] == ["https://cdn.example.com/x.jpg"]
    assert len(m.image_paths) == 1 and Path(m.image_paths[0]).read_bytes() == _png_bytes()
    assert m.html_path and "<p>Hi</p>" in Path(m.html_path).read_text()
    assert m.meta["infra"] == {
        "ips": ["192.0.2.1", "192.0.2.2"],
        "asns": ["AS13335 CLOUDFLARENET"],
        "favicon_mmh3": 123,
    }
    assert m.meta["extra_field_nested"] is True
    assert m.label == "K7" and m.source == SourceKind.RECORD
    assert ms[1].url == "https://tylko-adres.test/x" and ms[1].source == SourceKind.URL
    # ponowne wczytanie daje te same ścieżki plików pochodnych
    again = load_dataset(tmp_path / "records.jsonl")
    assert again[0].image_paths == m.image_paths and again[0].html_path == m.html_path


def test_ids_generated_stable_and_deduplicated(tmp_path: Path) -> None:
    recs = [
        {"text": "ta sama reklama"},
        {"text": "ta sama reklama"},
        {"id": "a", "text": "x"},
        {"id": "a", "text": "y"},
        {"id": 7.0, "text": "z"},
    ]
    (tmp_path / "r.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    ids = [m.id for m in load_dataset(tmp_path / "r.jsonl")]
    assert ids[0].startswith("m-") and len(ids[0]) == 14
    assert ids[1] == ids[0] + "-2"  # dokładny duplikat zostaje osobnym materiałem
    assert ids[2:] == ["a", "a-2", "7"]
    assert [m.id for m in load_dataset(tmp_path / "r.jsonl")] == ids


def test_xlsx_title_row_hyperlinks_merged_cells_embedded_image(tmp_path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage

    wb = Workbook()
    ws = wb.active
    ws.title = "Reklamy"
    ws.append(["Zestawienie reklam — NASK"])
    ws.append([])
    ws.append(["Lp.", "Treść", "Link do reklamy", "Kampania", "Data"])
    ws.append([1, "Zarabiaj 5000 zł", "reklama", "K1", "21.11.2026"])
    ws.append([2, "Inna treść", "https://b.test/x", None, datetime(2026, 11, 20, 8, 0)])
    ws["C4"].hyperlink = "https://a.test/landing"
    ws.merge_cells("D4:D5")
    ws.add_image(XLImage(io.BytesIO(_png_bytes("green"))), "F4")
    legend = wb.create_sheet("Legenda")
    legend.append(["Kod", "Opis"])
    legend.append(["K1", "kampania inwestycyjna"])
    wb.save(tmp_path / "zestawienie.xlsx")

    ms = _by_id(load_dataset(tmp_path / "zestawienie.xlsx"))
    assert set(ms) == {"1", "2"}  # arkusz „Legenda” pominięty
    assert ms["1"].url == "https://a.test/landing"  # cel hiperłącza zamiast tekstu „reklama”
    assert ms["1"].label == "K1" and ms["2"].label == "K1"  # scalona komórka
    assert ms["2"].observed_at == datetime(2026, 11, 20, 8, 0)
    assert len(ms["1"].image_paths) == 1 and Path(ms["1"].image_paths[0]).is_file()
    assert ms["2"].image_paths == []
    assert ms["1"].meta["source_row"] == 4


def test_headerless_csv_of_urls_and_url_list_txt(tmp_path: Path) -> None:
    (tmp_path / "lista.csv").write_text("https://a.test/1\nhttps://b.test/2\nhttps://c.test/3\n")
    ms = load_dataset(tmp_path / "lista.csv")
    assert [m.url for m in ms] == ["https://a.test/1", "https://b.test/2", "https://c.test/3"]
    (tmp_path / "links.txt").write_text("# zgłoszenia\nhxxp://scam[.]test/a\nwww.drugi.test\n\n")
    ms = load_dataset(tmp_path / "links.txt")
    assert [m.url for m in ms] == ["http://scam.test/a", "http://www.drugi.test"]
    assert all(m.source == SourceKind.URL and m.id.startswith("m-") for m in ms)


# ---------------------------------------------------------------- katalogi


def test_directory_without_records_file(tmp_path: Path) -> None:
    root = tmp_path / "zbior"
    _png(root / "a.png")
    _png(root / "b.jpg", "blue")
    (root / "page.html").write_text("<html><body>strona</body></html>")
    (root / "notatka.txt").write_text("Pilne! Zarabiaj z platformą X.")
    (root / "urls.txt").write_text("https://u1.test/a\nhttps://u2.test/b\n")
    (root / "README.md").write_text("opis zbioru")
    # para obraz + opis JSON o tej samej nazwie
    _png(root / "x1.png", "green")
    (root / "x1.json").write_text(json.dumps({"url": "https://x1.test", "kampania": "KX"}))
    # podkatalog = jeden materiał złożony
    bundle = root / "ad_17"
    _png(bundle / "screenshot.png", "yellow")
    (bundle / "landing.html").write_text("<html><body>lp</body></html>")
    (bundle / "info.json").write_text(json.dumps({"url": "https://ad17.test", "tekst": "opis", "campaign": "K2"}))
    # podkatalog z samymi obrazami = kolekcja
    for i in range(3):
        _png(root / "galeria" / f"g{i}.png", "white")
    # zapisana strona z katalogiem zasobów (pomijany)
    (root / "zapis.html").write_text("<html><body>zapis</body></html>")
    _png(root / "zapis_files" / "logo.png")
    # strona MHTML (Chrome „zapisz jako jeden plik”)
    (root / "strona.mhtml").write_text(
        "From: <Saved by Blink>\nSnapshot-Content-Location: https://mhtml.test/lp\nSubject: x\nMIME-Version: 1.0\n"
        'Content-Type: multipart/related; type="text/html"; boundary="B"\n\n--B\nContent-Type: text/html\n'
        "Content-Location: https://mhtml.test/lp\n\n<html><body>z mhtml</body></html>\n--B--\n"
    )

    ms = _by_id(load_dataset(root))
    assert ms["a"].source == SourceKind.IMAGE and ms["b"].source == SourceKind.IMAGE
    assert ms["page"].source == SourceKind.HTML
    assert ms["notatka"].source == SourceKind.TEXT and "platformą X" in (ms["notatka"].text or "")
    assert sorted(m.url for m in ms.values() if m.source == SourceKind.URL and m.id.startswith("m-")) == [
        "https://u1.test/a",
        "https://u2.test/b",
    ]
    assert "README" not in ms
    x1 = ms["x1"]
    assert x1.url == "https://x1.test" and x1.label == "KX" and len(x1.image_paths) == 1
    ad = ms["ad_17"]
    assert ad.source == SourceKind.RECORD and ad.label == "K2" and ad.text == "opis"
    assert ad.url == "https://ad17.test" and ad.html_path and len(ad.image_paths) == 1
    assert {"galeria/g0", "galeria/g1", "galeria/g2"} <= set(ms)
    assert not any("logo" in p for m in ms.values() for p in m.image_paths)
    mh = ms["strona"]
    assert mh.url == "https://mhtml.test/lp" and mh.html_path and "z mhtml" in Path(mh.html_path).read_text()
    for m in ms.values():
        assert all(Path(p).is_absolute() for p in m.image_paths)


def test_directory_prefers_records_file_with_most_rows_and_matches_ids(tmp_path: Path) -> None:
    (tmp_path / "small.csv").write_text("id,url\n1,https://a.test\n2,https://b.test\n")
    recs = [{"record_id": f"ad{i}", "tekst": f"reklama {i}", "kampania": "K"} for i in range(5)]
    (tmp_path / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    _png(tmp_path / "images" / "ad3.png")
    _png(tmp_path / "ad4_1.jpg")
    _png(tmp_path / "ad4_2.jpg", "blue")
    ms = _by_id(load_dataset(tmp_path))
    assert set(ms) == {f"ad{i}" for i in range(5)}
    assert len(ms["ad3"].image_paths) == 1 and ms["ad3"].source == SourceKind.RECORD
    assert len(ms["ad4"].image_paths) == 2
    assert ms["ad0"].image_paths == [] and ms["ad0"].source == SourceKind.TEXT


def test_label_from_dir(tmp_path: Path) -> None:
    _png(tmp_path / "kampania_A" / "1.png")
    _png(tmp_path / "kampania_A" / "2.png", "blue")
    _png(tmp_path / "kampania_B" / "3.png", "green")
    _png(tmp_path / "kampania_B" / "4.png", "white")
    ms = load_dataset(tmp_path, label_from_dir=True)
    assert {m.id: m.label for m in ms} == {
        "kampania_A/1": "kampania_A",
        "kampania_A/2": "kampania_A",
        "kampania_B/3": "kampania_B",
        "kampania_B/4": "kampania_B",
    }
    assert all(m.label is None for m in load_dataset(tmp_path))


def test_eml_file(tmp_path: Path) -> None:
    (tmp_path / "zgloszenie.eml").write_text(
        "From: Kurier <kurier@paczka.test>\nTo: ofiara@example.com\nSubject: Paczka wstrzymana\n"
        "Date: Fri, 20 Nov 2026 10:00:00 +0100\nContent-Type: text/plain; charset=utf-8\n\n"
        "Dopłać 1,99 zł: https://paczka.test/pay\n"
    )
    (m,) = load_dataset(tmp_path / "zgloszenie.eml")
    assert m.source == SourceKind.TEXT
    assert "Paczka wstrzymana" in (m.text or "") and "https://paczka.test/pay" in (m.text or "")
    assert m.advertiser and "kurier@paczka.test" in m.advertiser
    assert m.observed_at == datetime(2026, 11, 20, 9, 0)


def test_bad_inputs_do_not_crash(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nie-ma")
    (tmp_path / "zly.json").write_text("{[[[")
    assert load_dataset(tmp_path / "zly.json") == []
    (tmp_path / "pusty.csv").write_text("")
    assert load_dataset(tmp_path / "pusty.csv") == []
    (tmp_path / "puste_pola.jsonl").write_text(json.dumps({"id": "x", "kategoria": "inne"}))
    (m,) = load_dataset(tmp_path / "puste_pola.jsonl")
    assert m.source == SourceKind.RECORD and any("bez treści" in w for w in m.meta["ingest_warnings"])


# ---------------------------------------------------------------- nowy materiał i zapis


def test_material_from_input(tmp_path: Path) -> None:
    img = _png(tmp_path / "zrzut.png")
    m = material_from_input(url="scam.test/x", image_path=img, text="  Zarabiaj  ")
    assert m.id.startswith("new-") and len(m.id) == 14
    assert m.url == "http://scam.test/x" and m.text == "Zarabiaj"
    assert m.image_paths == [str(img.resolve())] and m.source == SourceKind.RECORD
    assert material_from_input(url="scam.test/x", image_path=img, text="Zarabiaj").id == m.id
    assert material_from_input(url="https://a.test").source == SourceKind.URL
    assert material_from_input(image_path=str(img)).source == SourceKind.IMAGE
    assert material_from_input(text="tylko tekst").source == SourceKind.TEXT
    empty = material_from_input()
    assert empty.id.startswith("new-") and "brak danych wejściowych" in empty.meta["ingest_warnings"]
    missing = material_from_input(image_path=tmp_path / "nie-ma.png")
    assert any("brak pliku obrazu" in w for w in missing.meta["ingest_warnings"])


def test_workspace_roundtrip_and_reload_materials_jsonl(tmp_path: Path) -> None:
    _png(tmp_path / "d" / "images" / "1.png")
    (tmp_path / "d" / "r.jsonl").write_text(
        json.dumps(
            {
                "id": 1,
                "url": "https://a.test",
                "image": "1.png",
                "date": 1763740800,
                "campaign": "K",
                "infra": {"ips": ["192.0.2.1"]},
                "liczba": float("nan"),
                "lista": [1, 2],
            }
        )
    )
    ms = load_dataset(tmp_path / "d" / "r.jsonl")
    ws = Workspace(tmp_path / "ws")
    ws.save_materials(ms)
    again = Workspace(tmp_path / "ws").materials()
    assert [m.model_dump() for m in again] == [m.model_dump() for m in ms]
    # materials.jsonl z katalogu roboczego też da się wczytać jako zbiór; zdjęcia leżą poza jego katalogiem,
    # więc zgodnie z regułą bezpieczeństwa (ścieżki tylko w katalogu wczytywanego zbioru) są odrzucone z ostrzeżeniem
    reloaded = load_dataset(tmp_path / "ws" / "materials.jsonl")
    r, o = reloaded[0], ms[0]
    assert (r.id, r.url, r.label, r.observed_at) == (o.id, o.url, o.label, o.observed_at)
    assert r.image_paths == [] and any("spoza katalogu" in w for w in r.meta.get("ingest_warnings", []))
    # bez zdjęcia zostaje sam adres, więc rodzaj materiału to „url” zamiast „record”
    assert r.meta["infra"] == o.meta["infra"] and r.source.value == "url"


def test_sidecars_crlf_csv_and_content_sniffing(tmp_path: Path) -> None:
    # opisy „obraz.png.json” + jeden opis z dwoma rekordami nie zamienia katalogu w plik rekordów
    d = tmp_path / "zrzuty"
    for i in range(4):
        _png(d / f"ad{i}.png", "red" if i % 2 else "blue")
        (d / f"ad{i}.png.json").write_text(json.dumps({"url": f"https://ad{i}.test", "campaign": f"K{i}"}))
    (d / "extra.json").write_text(json.dumps([{"url": "https://e1.test"}, {"url": "https://e2.test"}]))
    ms = _by_id(load_dataset(d))
    assert {f"ad{i}" for i in range(4)} <= set(ms)
    assert ms["ad1"].url == "https://ad1.test" and ms["ad1"].label == "K1" and len(ms["ad1"].image_paths) == 1
    assert {m.url for m in ms.values()} >= {"https://e1.test", "https://e2.test"}

    # CSV z końcami linii Windows i nieznanymi nazwami kolumn — rozpoznanie po zawartości
    _png(tmp_path / "s1.png")
    (tmp_path / "x.csv").write_bytes(
        b'kol_a,kol_b,kol_c\r\nhttps://a.test/x,s1.png,"Bardzo dlugi opis reklamy, ktory na pewno jest tekstem i ma spacje"\r\n'
    )
    (m,) = load_dataset(tmp_path / "x.csv")
    assert m.url == "https://a.test/x" and len(m.image_paths) == 1 and m.text and m.text.startswith("Bardzo")

    # lista obrazów w JSON pod nieznaną nazwą pola
    (tmp_path / "y.json").write_text(json.dumps([{"screens": ["s1.png"], "opis": "x"}, {"screens": [], "opis": "y"}]))
    first = load_dataset(tmp_path / "y.json")[0]
    assert len(first.image_paths) == 1


def test_unreadable_single_file_gives_clear_error(tmp_path: Path) -> None:
    (tmp_path / "zepsuty.xlsx").write_bytes(b"to nie jest plik xlsx")
    with pytest.raises(ValueError, match="zepsuty.xlsx"):
        load_dataset(tmp_path / "zepsuty.xlsx")
    # ten sam plik w katalogu nie przerywa wczytywania pozostałych materiałów
    _png(tmp_path / "ok.png")
    ms = load_dataset(tmp_path)
    assert [m.id for m in ms] == ["ok"]
