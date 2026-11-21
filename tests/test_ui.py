"""Testy interfejsu: czyste funkcje (components, graph) + test dymny aplikacji Streamlit (AppTest).

Bez sieci; moduły innych zespołów (pipeline, ingest, export, eval) są zastępowane atrapami.
"""

from __future__ import annotations

import json
import re
import sys
import types
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:  # uruchomienie przez `.venv/bin/pytest` bez instalacji pakietu
    sys.path.insert(0, str(_REPO))

from slad.models import (  # noqa: E402
    Assignment,
    AssignmentStatus,
    Campaign,
    CampaignSummary,
    Evidence,
    Face,
    FeatureSet,
    ImageFeatures,
    InfraFeatures,
    Link,
    Material,
    Relation,
    SourceKind,
    TextFeatures,
)
from slad.store import Workspace  # noqa: E402
from slad.ui import components as C  # noqa: E402
from slad.ui import graph as G  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "slad" / "ui" / "app.py"

# ---------------------------------------------------------------- dane testowe


def ev(
    kind: str, group: str, detail: str, weight: float, value: str | None = None, sim: float | None = None
) -> Evidence:
    return Evidence(kind=kind, group=group, detail=detail, value=value, similarity=sim, weight=weight)


def _png(path: Path, color: str) -> Path:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (160, 120), color)
    ImageDraw.Draw(im).rectangle([20, 20, 90, 80], outline="white", width=4)
    im.save(path)
    return path


def _image(path: Path, faces: list[Face] | None = None) -> ImageFeatures:
    return ImageFeatures(
        path=str(path),
        sha256="ab" * 32,
        width=160,
        height=120,
        phash="f0f0f0f0f0f0f0f0",
        dhash="0f0f0f0f0f0f0f0f",
        whash="ffff0000ffff0000",
        colorhash="07000000000",
        dominant_colors=["#2a78d6", "#ffffff", "not-a-colour"],
        faces=faces or [],
    )


LINK_12 = Link(
    a="m1",
    b="m2",
    score=14.5,
    relation=Relation.DUPLICATE,
    coverage=0.75,
    evidence=[
        ev("person", "osoby", "ta sama osoba: Jan Kowalski", 2.0, value="Jan Kowalski"),
        ev(
            "tracker_id",
            "infrastruktura",
            "wspólny identyfikator Google Tag Manager GTM-ABC123",
            6.0,
            value="GTM-ABC123",
        ),
        ev("text_minhash", "tekst", "niemal identyczny tekst", 4.5, sim=0.91),
        ev("domain", "infrastruktura", "ta sama domena", 2.0, value="zysk-teraz.example"),
    ],
)
LINK_23 = Link(
    a="m2",
    b="m3",
    score=4.2,
    relation=Relation.RELATED,
    coverage=0.33,
    evidence=[
        ev("narrative", "narracja", "ta sama narracja: inwestycje", 1.2, value="inwestycje"),
        ev("keyword", "tekst", "wspólne słowa", 3.0),
    ],
)


def build_workspace(root: Path) -> Workspace:
    img_dir = root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    a = _png(img_dir / "a.png", "#2a78d6")
    b = _png(img_dir / "b.png", "#1baf7a")
    materials = [
        Material(
            id="m1",
            source=SourceKind.URL,
            url="https://zysk-teraz.example/promo",
            text="Jan Kowalski poleca: zarabiaj 5000 zł dziennie!",
            image_paths=[str(a)],
            platform="facebook",
            observed_at=datetime(2026, 11, 20, 12, 0),
            label="kampania-A",
        ),
        Material(
            id="m2",
            source=SourceKind.IMAGE,
            url="https://zysk-teraz.example/start",
            image_paths=[str(b)],
            label="kampania-A",
        ),
        Material(
            id="m3",
            source=SourceKind.TEXT,
            text="Paczka czeka na dopłatę <script>alert(1)</script> [kliknij](http://evil.example)",
            label="kampania-B",
        ),
    ]
    ws = Workspace(root)
    ws.save_materials(materials)
    face = Face(bbox=(30, 25, 50, 50), score=0.97, identity="Jan Kowalski", identity_score=0.71)
    infra = InfraFeatures(
        input_urls=["https://zysk-teraz.example/promo"],
        redirect_chain=["https://bit.example/x", "https://zysk-teraz.example/promo"],
        final_url="https://zysk-teraz.example/promo",
        http_status=200,
        domains=["zysk-teraz.example"],
        tracker_ids=["gtm:GTM-ABC123"],
        ips=["203.0.113.7"],
        asns=["AS64500 EXAMPLE"],
        favicon_mmh3=-123456,
        registrar="Example Registrar",
        domain_created=datetime(2026, 11, 14),
        fetched_at=datetime(2026, 11, 20),
        errors=["rdap: timeout"],
    )
    ws.save_features(
        FeatureSet(
            material_id="m1",
            text=TextFeatures(
                input_text=materials[0].text or "",
                ocr_text="ZARABIAJ 5000 ZŁ",
                text="jan kowalski poleca zarabiaj",
                persons=["Jan Kowalski"],
                brands=["TVN24"],
                narratives=["inwestycje"],
                promise="5000 zł dziennie",
                cta="Zarejestruj się",
                iocs={"phone": ["+48600000000"], "crypto_btc": ["bc1qexample"]},
                minhash=[1, 2, 3],
            ),
            images=[_image(a, [face])],
            infra=infra,
            available=["tekst", "obraz", "twarze", "infrastruktura"],
        )
    )
    ws.save_features(
        FeatureSet(
            material_id="m2",
            text=TextFeatures(text="kowalski inwestycja", persons=["Jan Kowalski"], narratives=["inwestycje"]),
            images=[_image(b)],
            infra=InfraFeatures(domains=["zysk-teraz.example"], tracker_ids=["gtm:GTM-ABC123"], ips=["203.0.113.7"]),
            available=["tekst", "obraz", "infrastruktura"],
        )
    )
    ws.save_features(
        FeatureSet(
            material_id="m3",
            text=TextFeatures(
                input_text=materials[2].text or "", text="paczka czeka na dopłatę", narratives=["inwestycje"]
            ),
            available=["tekst"],
            errors=["infrastruktura: brak adresu URL"],
        )
    )
    ws.save_links([LINK_12, LINK_23])
    ws.save_campaigns(
        [
            Campaign(
                id="K001",
                name="Jan Kowalski · inwestycje · zysk-teraz.example",
                members=["m1", "m2"],
                cohesion=14.5,
                summary=CampaignSummary(
                    persons=[("Jan Kowalski", 2)],
                    brands=[("TVN24", 1)],
                    narratives=[("inwestycje", 2)],
                    domains=[("zysk-teraz.example", 2)],
                    indicators=[("tracker_id:gtm:GTM-ABC123", 2)],
                    keywords=["zysk", "krypto"],
                    first_seen=datetime(2026, 11, 19),
                    last_seen=datetime(2026, 11, 20),
                ),
                key_evidence=[LINK_12.evidence[1], LINK_12.evidence[0]],
            )
        ]
    )
    return ws


@pytest.fixture()
def bundle(tmp_path: Path) -> C.Bundle:
    build_workspace(tmp_path / "ws")
    return C.load_bundle(tmp_path / "ws")


# ---------------------------------------------------------------- components


def test_evidence_frame_columns_sorted_and_formatted() -> None:
    df = C.evidence_frame(LINK_12.evidence)
    assert list(df.columns) == ["Grupa", "Przesłanka", "Wartość", "Waga"]
    assert df["Waga"].tolist() == sorted(df["Waga"].tolist(), reverse=True)
    assert df.iloc[0]["Grupa"] == "🌐 infrastruktura"
    assert df.iloc[0]["Wartość"] == "GTM-ABC123"
    sim_row = df[df["Przesłanka"] == "niemal identyczny tekst"].iloc[0]
    assert sim_row["Wartość"] == "podobieństwo 91%"
    assert len(C.evidence_frame(LINK_12.evidence, limit=2)) == 2
    empty = C.evidence_frame([])
    assert empty.empty and list(empty.columns) == C.EVIDENCE_COLUMNS


def test_style_groups_colours_group_cells() -> None:
    df = C.evidence_frame(LINK_12.evidence)
    html = C.style_groups(df).to_html()
    assert C.tint(C.GROUP_COLORS["infrastruktura"], 0.22) in html
    assert C.tint(C.GROUP_COLORS["osoby"], 0.22) in html
    assert C.group_cell_style("nieznana") == ""
    assert C.style_groups(pd.DataFrame()) is not None


def test_evidence_html_ordered_coloured_and_escaped() -> None:
    evil = ev("title", "infrastruktura", "tytuł <script>alert(1)</script>", 9.0, value="<img src=x onerror=alert(1)>")
    html = C.evidence_html([*LINK_12.evidence, evil])
    assert [h for h in ("Grupa", "Przesłanka", "Wartość", "Waga") if f"<th>{h}</th>" in html] == C.EVIDENCE_COLUMNS
    assert "<script>" not in html and "<img" not in html and "&lt;script&gt;" in html
    weights = [float(w) for w in re.findall(r"</span></span>([0-9.]+)</td>", html)]
    assert weights == sorted(weights, reverse=True) and weights[0] == 9.0
    assert C.tint(C.GROUP_COLORS["tekst"], 0.22) in html and "width:100%" in html
    assert C.evidence_html([]) == "" and C.evidence_html(LINK_12.evidence, limit=1).count("<tr>") == 2


def test_tint_and_hex_validation() -> None:
    assert C.tint("#000000", 0.5) == "#808080"
    assert C.tint("#2a78d6", 0.0) == "#ffffff"
    assert C.tint("javascript:alert(1)") == "#ffffff"
    assert C.is_hex_color("#a1b2c3") and not C.is_hex_color("red")


def test_campaign_styles_stable_under_filtering() -> None:
    camps = [Campaign(id=f"K{i:03d}", name=f"k{i}", members=[f"m{i}"]) for i in range(1, 11)]
    full = C.campaign_styles(camps)
    subset = C.campaign_styles(list(reversed(camps)))
    assert full == subset
    assert full["K001"] == (C.CATEGORICAL[0], "dot")
    assert full["K009"] == (C.CATEGORICAL[0], "square")  # 9. kampania: ten sam kolor, inny kształt
    assert len({full[c.id] for c in camps}) == 10


def test_md_escape_neutralises_markdown_and_html() -> None:
    s = C.md_escape("[kliknij](http://evil.example) ![x](http://t.example/p.png) <b>x</b> **pogrubienie**")
    assert "](" not in s.replace("\\](", "").replace("\\]\\(", "")
    assert "\\[" in s and "\\<" in s and "\\*" in s
    safe = C.md_safe("Pasuje do **K001**: https://evil.example i www.evil.example [x](y)")
    assert "**K001**" in safe
    assert "https://" not in safe and "www.evil" not in safe
    assert "\\[" in safe
    # GFM linkuje adresy i e-maile także po escapowaniu — muszą być rozbite
    raw = C.md_escape("https://evil.example/x www.evil.example kontakt@evil.example")
    assert "://" not in raw.replace("\\", "") and "www." not in raw.replace("\\", "")
    assert "@evil" not in raw


def test_defang_urls() -> None:
    assert C.defang("https://zysk-teraz.example/promo?a=b.c") == "hxxps://zysk-teraz[.]example/promo?a=b.c"
    assert C.defang("http://1.2.3.4:8080/x") == "hxxp://1[.]2[.]3[.]4:8080/x"
    assert C.defang("bit.example/abc") == "bit[.]example/abc"
    assert C.defang(None) == "" and C.defang("  ") == ""


def test_run_summary(bundle: C.Bundle) -> None:
    info = {
        "dataset": "/data/nask/reklamy.csv",
        "materials": 120,
        "time_s": 33.6,
        "errors": 2,
        "error_samples": ["m7: OCR: nieczytelny plik"],
        "link_error": "ValueError: x",
        "extracted": 100,
        "cached": 20,
        "finished_at": "2026-11-21T19:58:12+00:00",
        "settings": {"network": False, "ocr": True, "llm": True, "faces": False, "llm_model": "qwen"},
    }
    sentence, details, samples = C.run_summary(info)
    assert sentence.startswith("Ostatnie przetwarzanie: 2026-11-21 19:58 UTC")
    assert "reklamy.csv" in sentence and "120 materiałów w 34 s" in sentence and "błędami: 2" in sentence
    rows = dict(zip(details["Cecha"], details["Wartość"], strict=True))
    assert rows["Sieć / OCR / LLM / twarze"] == "nie / tak / tak / nie" and rows["Model LLM"] == "qwen"
    assert samples[0] == "łączenie: ValueError: x" and len(samples) == 2
    assert C.run_summary(None)[0] == "" and bundle.run_info is None


def test_overview_counts_and_coverage(bundle: C.Bundle) -> None:
    k = C.overview_counts(bundle)
    assert k["materials"] == 3 and k["campaigns"] == 1 and k["links"] == 2
    assert k["duplikat"] == 1 and k["powiązany"] == 1 and k["wariant"] == 0
    assert k["unlinked"] == 0 and k["unassigned"] == 1 and k["labeled"] == 3
    cov = C.coverage_frame(bundle).set_index("Grupa cech")
    assert cov.loc["📝 Tekst", "Materiały"] == 3
    assert cov.loc["🙂 Twarze", "Materiały"] == 1
    assert cov.loc["🌐 Infrastruktura", "Udział"] == pytest.approx(66.7)


def test_top_values_counts_materials_once(bundle: C.Bundle) -> None:
    persons = C.top_values(bundle.features.values(), "persons")
    assert persons.iloc[0].tolist() == ["Jan Kowalski", 2]
    domains = C.top_values(bundle.features.values(), "domains")
    assert domains.iloc[0].tolist() == ["zysk-teraz.example", 2]
    assert list(C.top_values([], "brands").columns) == ["Marka / instytucja", "Materiały"]


def test_campaigns_frame_and_summary_tables(bundle: C.Bundle) -> None:
    df = C.campaigns_frame(bundle.campaigns)
    row = df.iloc[0]
    assert row["ID"] == "K001" and row["Materiały"] == 2
    assert row["Główna osoba"] == "Jan Kowalski" and row["Główna domena"] == "zysk-teraz.example"
    ind = C.pairs_frame(bundle.campaigns[0].summary.indicators, "Wskaźnik", C.pretty_indicator)
    assert ind.iloc[0]["Wskaźnik"] == "tracker: gtm:GTM-ABC123"
    assert C.campaign_links(bundle, bundle.campaigns[0]) == [LINK_12]


def test_bundle_indexes(bundle: C.Bundle) -> None:
    assert bundle.campaign_of == {"m1": "K001", "m2": "K001"}
    assert [lk.score for lk in bundle.links_by_material["m2"]] == [14.5, 4.2]
    assert bundle.label("m1").startswith("m1 · zysk-teraz.example")
    assert bundle.campaign_label(None) == "bez kampanii"
    assert bundle.campaign_label("K001").startswith("K001 · Jan Kowalski")


def test_load_bundle_tolerates_corrupt_files(tmp_path: Path) -> None:
    ws = build_workspace(tmp_path / "ws")
    with ws.path("links.jsonl").open("a", encoding="utf-8") as f:
        f.write("{to nie jest json\n")
    ws.path("stats.json").write_text("[]", encoding="utf-8")
    (ws.root / "features" / "m3.json").write_text("{zepsute", encoding="utf-8")
    b = C.load_bundle(ws.root)
    assert len(b.materials) == 3 and len(b.links) == 2
    assert "m3" not in b.features
    assert any("links.jsonl" in e for e in b.errors)
    assert any("stats.json" in e for e in b.errors)
    assert any("cechy" in e for e in b.errors)
    empty = C.load_bundle(tmp_path / "pusty")
    assert not empty.materials and not empty.errors


def test_workspace_signature_and_listing(tmp_path: Path) -> None:
    ws = build_workspace(tmp_path / "workspace" / "demo")
    sig1 = C.workspace_signature(ws.root)
    ws.save_links([LINK_12])
    assert C.workspace_signature(ws.root) != sig1
    (tmp_path / "workspace" / "inny").mkdir()
    (tmp_path / "workspace" / ".ukryty").mkdir()
    extra = tmp_path / "gdzie-indziej"
    found = C.list_workspaces(tmp_path / "workspace", [extra])
    assert found[0] == extra.resolve()
    assert [p.name for p in found[1:]] == ["demo", "inny"]
    assert C.workspace_label(ws.root, tmp_path) == "workspace/demo"


def test_link_explanations() -> None:
    contrib = C.group_contributions(LINK_12.evidence)
    assert contrib[0][0] == "infrastruktura" and contrib[0][1] == pytest.approx(8.0)
    assert sum(s for _, _, s in contrib) == pytest.approx(1.0)
    sentence = C.link_sentence(LINK_12)
    assert "14.5" in sentence and "duplikat" in sentence and "infrastruktura" in sentence
    agg = C.aggregate_evidence([LINK_12, LINK_23])
    assert list(agg.columns) == ["Grupa", "Przesłanka", "Wartość", "Powiązania", "Łączna waga"]
    assert agg.iloc[0]["Łączna waga"] == 6.0
    nf = C.nearest_frame([LINK_12], "m2")
    assert nf.iloc[0]["Materiał"] == "m1" and nf.iloc[0]["Relacja"] == "duplikat"
    lf = C.links_frame([LINK_23, LINK_12])
    assert lf["Wynik"].tolist() == [14.5, 4.2]


def test_material_tables(bundle: C.Bundle) -> None:
    fs1 = bundle.features["m1"]
    facts = C.text_facts(fs1).set_index("Cecha")["Wartość"]
    assert facts["Osoby"] == "Jan Kowalski" and facts["Obietnica"] == "5000 zł dziennie"
    iocs = C.iocs_frame(fs1.text.iocs)
    assert set(iocs["Typ"]) == {"telefon", "portfel BTC"}
    infra = C.infra_frame(fs1.infra).set_index("Cecha")["Wartość"]
    assert infra["Trackery"] == "gtm:GTM-ABC123"
    assert infra["Data rejestracji domeny"] == "2026-11-14 (domena miała 6 dni)"
    assert C.redirect_frame(fs1.infra)["Krok"].tolist() == [1, 2]
    faces = C.faces_frame(fs1.images)
    assert faces.iloc[0]["Tożsamość"] == "Jan Kowalski"
    assert C.image_frame(fs1.images).iloc[0]["Twarze"] == 1
    summary = C.features_summary(fs1).set_index("Cecha")["Wartość"]
    assert "portfel BTC: 1" in summary["Wskaźniki IOC"]
    assert C.all_errors(fs1) == ["rdap: timeout"]
    assert C.infra_frame(InfraFeatures()).empty


def test_thumbnails_and_face_overlay(bundle: C.Bundle, tmp_path: Path) -> None:
    img = C.first_image(bundle.materials["m1"], bundle.features["m1"])
    assert img is not None
    assert C.thumbnail_bytes(img, 64)[:2] == b"\xff\xd8"
    assert C.faces_overlay(img, bundle.features["m1"].images[0].faces)[:4] == b"\x89PNG"
    bad = tmp_path / "zly.png"
    bad.write_bytes(b"to nie obraz")
    assert C.thumbnail_bytes(bad) is None
    assert C.thumbnail_bytes(tmp_path / "brak.png") is None
    assert (
        C.first_image(Material(id="x", source=SourceKind.IMAGE, image_paths=[str(bad.with_name("nie-ma.png"))])) is None
    )


@pytest.mark.parametrize(
    ("args", "kwargs", "expected"),
    [
        ((3, 10), {}, (0.3, "3/10")),
        ((5, 10, "OCR"), {}, (0.5, "OCR (5/10)")),
        (("cechy", 1, 4), {}, (0.25, "cechy (1/4)")),
        ((0.4,), {}, (0.4, "")),
        ((40,), {}, (0.4, "")),
        ((), {"done": 2, "total": 8, "stage": "powiązania"}, (0.25, "powiązania (2/8)")),
        ((), {"fraction": 1.7, "message": "koniec"}, (1.0, "koniec")),
        ((1, 0), {}, (0.0, "")),
        (("start",), {}, (0.0, "start")),
    ],
)
def test_parse_progress(args: tuple, kwargs: dict, expected: tuple[float, str]) -> None:
    frac, text = C.parse_progress(*args, **kwargs)
    assert frac == pytest.approx(expected[0])
    assert text == expected[1]


def test_fallback_material_and_safe_filename(tmp_path: Path) -> None:
    img = _png(tmp_path / "zrzut.png", "#eda100")
    m = C.fallback_material(url=None, image_path=img, text="tekst")
    assert m.source == SourceKind.IMAGE and m.image_paths == [str(img.resolve())] and m.id.startswith("nowy-")
    assert C.fallback_material(url="https://a.example").source == SourceKind.URL
    assert C.fallback_material(text="x").id == C.fallback_material(text="x").id
    assert C.safe_filename("../../etc/passwd") == "etc_passwd"
    assert C.safe_filename("") == "plik"


def test_report_tables_flattens_nested_report() -> None:
    report = {
        "n": 3,
        "pairwise": {"precision": 0.9, "recall": 0.75, "f1": 0.8181818},
        "per_campaign": {"K001": {"size": 2, "purity": 1.0}, "K002": {"size": 1, "purity": 0.5}},
        "assign": [{"material": "m1", "ok": True}],
        "notes": ["a", "b"],
    }
    scalars, tables = C.report_tables(report)
    metrics = dict(zip(scalars["Metryka"], scalars["Wartość"], strict=True))
    assert metrics["n"] == "3" and metrics["pairwise.f1"] == "0.8182" and metrics["notes"] == "a, b"
    titles = [t for t, _ in tables]
    assert titles == ["per_campaign", "assign"]
    assert tables[0][1].loc["K002", "purity"] == 0.5


def test_settings_tables_hide_secrets() -> None:
    from slad.settings import load_settings

    cfg = load_settings(llm={"api_key": "SEKRET-123", "model": "qwen"})
    w = C.weights_frame(cfg.link.weights)
    assert w.iloc[0]["Klucz"] == "crypto_wallet" and w["Waga"].is_monotonic_decreasing
    assert set(C.thresholds_frame(cfg.link.thresholds)["Relacja"]) == {"duplikat", "wariant", "powiązany"}
    st_df = C.settings_frame(cfg)
    assert "SEKRET-123" not in st_df.to_csv()
    assert "qwen" in st_df.to_csv()


def test_zip_helpers(tmp_path: Path, bundle: C.Bundle) -> None:
    import io
    import zipfile

    (tmp_path / "out" / "sub").mkdir(parents=True)
    (tmp_path / "out" / "a.csv").write_text("x", encoding="utf-8")
    (tmp_path / "out" / "sub" / "b.cypher").write_text("y", encoding="utf-8")
    names = zipfile.ZipFile(io.BytesIO(C.zip_dir(tmp_path / "out"))).namelist()
    assert names == ["a.csv", "sub/b.cypher"]
    z = zipfile.ZipFile(io.BytesIO(C.zip_frames({"przypisania.csv": C.assignments_frame(bundle)})))
    assert "m1,K001" in z.read("przypisania.csv").decode()


# ---------------------------------------------------------------- graph


def test_material_indicators(bundle: C.Bundle) -> None:
    ind = G.material_indicators(bundle.features["m1"])
    assert ind["tracker"] == {"gtm:GTM-ABC123"}
    assert ind["favicon"] == {"-123456"}
    assert ind["wallet"] == {"bc1qexample"} and ind["face"] == {"Jan Kowalski"}
    assert G.material_indicators(None) == {}


def test_build_graph_nodes_edges_and_filters(bundle: C.Bundle) -> None:
    b = bundle
    data = G.build_graph(b.materials, b.features, b.links, b.campaigns, styles=b.styles)
    nodes = {n["id"]: n for n in data.nodes}
    assert set(nodes) == {"m1", "m2", "m3"}
    assert nodes["m1"]["color"] == C.CATEGORICAL[0]
    assert nodes["m3"]["color"] == C.UNASSIGNED_COLOR
    assert "Kampania: K001" in nodes["m1"]["title"] and "Osoby: Jan Kowalski" in nodes["m1"]["title"]
    edges = {(e["from"], e["to"]): e for e in data.edges}
    dup, rel = edges[("m1", "m2")], edges[("m2", "m3")]
    assert dup["width"] > rel["width"]
    assert dup["color"]["color"] == C.RELATION_COLORS["duplikat"] and not dup["dashes"]
    assert rel["dashes"] is True
    assert dup["title"].splitlines()[3].startswith("• [infrastruktura]")  # przesłanki malejąco po wadze
    assert data.n_materials == 3 and data.n_links == 2

    strong = G.build_graph(b.materials, b.features, b.links, b.campaigns, min_score=10, include_isolated=False)
    assert {n["id"] for n in strong.nodes} == {"m1", "m2"} and strong.n_links == 1
    only_rel = G.build_graph(
        b.materials, b.features, b.links, b.campaigns, relations=["powiązany"], include_isolated=False
    )
    assert [(e["from"], e["to"]) for e in only_rel.edges] == [("m2", "m3")]
    camp = G.build_graph(b.materials, b.features, b.links, b.campaigns, campaign_id="K001")
    assert {n["id"] for n in camp.nodes} == {"m1", "m2"}
    camp_nb = G.build_graph(b.materials, b.features, b.links, b.campaigns, campaign_id="K001", include_neighbors=True)
    assert {n["id"] for n in camp_nb.nodes} == {"m1", "m2", "m3"}
    labels = [item["label"] for item in data.legend]
    assert any(lbl.startswith("K001 · ") for lbl in labels) and "bez kampanii (1)" in labels
    assert "duplikat (1)" in labels and "powiązany (1)" in labels


def test_build_graph_dedupes_and_truncates(bundle: C.Bundle) -> None:
    b = bundle
    dup = LINK_12.model_copy(update={"a": "m2", "b": "m1", "score": 20.0})
    data = G.build_graph(b.materials, b.features, [*b.links, dup], b.campaigns)
    assert data.n_links == 2
    assert max(e["width"] for e in data.edges) == pytest.approx(1 + 20 / 4)
    small = G.build_graph(b.materials, b.features, b.links, b.campaigns, max_edges=1)
    assert small.truncated and small.n_links == 1


def test_build_graph_indicator_hubs(bundle: C.Bundle) -> None:
    b = bundle
    data = G.build_graph(b.materials, b.features, b.links, b.campaigns, hubs=True)
    hubs = {n["id"]: n for n in data.nodes if n["id"].startswith("hub:")}
    assert "hub:tracker:gtm:GTM-ABC123" in hubs and "hub:domain:zysk-teraz.example" in hubs
    assert "hub:ip:203.0.113.7" in hubs
    assert all(n["shape"] == "diamond" for n in hubs.values())
    assert not any(k.startswith("hub:favicon") for k in hubs)  # favicona tylko w jednym materiale
    hub_edges = [e for e in data.edges if e["to"] == "hub:tracker:gtm:GTM-ABC123"]
    assert {e["from"] for e in hub_edges} == {"m1", "m2"}
    assert data.n_hubs == len(hubs) and any(item["type"] == "hub" for item in data.legend)
    only_ip = G.build_graph(b.materials, b.features, b.links, b.campaigns, hubs=True, hub_kinds=["ip"])
    assert [n["id"] for n in only_ip.nodes if n["id"].startswith("hub:")] == ["hub:ip:203.0.113.7"]


def test_tooltip_is_plain_and_bounded() -> None:
    t = G.tooltip(["<a href='x'>link</a>", "linia\x00", *[f"l{i}" for i in range(30)]], max_lines=5)
    assert "href" not in t
    assert len(t.splitlines()) == 5 and t.splitlines()[-1].startswith("…")
    assert "\x00" not in t


def test_render_html_offline_escaped_and_with_legend(bundle: C.Bundle) -> None:
    pytest.importorskip("pyvis")
    b = bundle
    evil = Campaign(id="K001", name="<script>alert('x')</script>", members=["m1", "m2"])
    m1 = b.materials["m1"].model_copy(update={"url": "https://x.example/?href=</script><script>alert(1)</script>"})
    data = G.build_graph({**b.materials, "m1": m1}, b.features, b.links, [evil], hubs=True)
    page = G.render_html(data, height=420)
    assert "vis.Network" in page and "new vis.DataSet" in page
    assert "cdn.jsdelivr.net/npm/bootstrap" not in page  # brak zależności od CDN
    assert 'id="slad-legend"' in page and "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in page
    assert "<script>alert(1)</script>" not in page and "<script>alert('x')</script>" not in page
    assert "popup.innerHTML" not in page  # podpowiedzi jako zwykły tekst
    assert "height: 420px" in page
    empty = G.render_html(G.GraphData())
    assert "Brak materiałów" in empty and "vis.Network" not in empty


def test_legend_html_rejects_bad_colours() -> None:
    html = G.legend_html([{"type": "campaign", "color": "red;background:url(x)", "label": "<b>K1</b>"}])
    assert "url(x)" not in html and "&lt;b&gt;K1&lt;/b&gt;" in html


# ---------------------------------------------------------------- aplikacja (AppTest)


def _apptest():
    testing = pytest.importorskip("streamlit.testing.v1")
    return testing.AppTest.from_file(str(APP), default_timeout=60)


def _install_fakes(monkeypatch: pytest.MonkeyPatch, calls: list) -> None:
    """Atrapy modułów innych zespołów (mogą jeszcze nie istnieć) — kod UI importuje je leniwie."""
    pipeline = types.ModuleType("slad.pipeline")

    def analyze_new(material: Material, ws: Workspace, cfg, add: bool = False):
        calls.append(("analyze", material.id, add, cfg.network.enabled))
        fs = FeatureSet(
            material_id=material.id,
            text=TextFeatures(text="jan kowalski", persons=["Jan Kowalski"], narratives=["inwestycje"]),
            available=["tekst"],
        )
        near = Link(
            a=material.id, b="m1", score=9.0, relation=Relation.VARIANT, coverage=0.5, evidence=LINK_12.evidence
        )
        assignment = Assignment(
            material_id=material.id,
            campaign_id="K001",
            score=9.0,
            status=AssignmentStatus.ASSIGNED,
            nearest=[near],
            explanation="Materiał pasuje do kampanii **K001**: wspólny tracker GTM-ABC123 i ta sama osoba.",
        )
        if add:
            ws.add_material(material)
            ws.save_features(fs)
        return fs, assignment

    def process_dataset(path, ws, cfg, progress=None):
        calls.append(("process", str(path)))
        if progress:
            progress(1, 2, "cechy")
            progress(2, 2, "powiązania")

    pipeline.analyze_new = analyze_new
    pipeline.process_dataset = process_dataset
    monkeypatch.setitem(sys.modules, "slad.pipeline", pipeline)

    ingest = types.ModuleType("slad.ingest")

    def material_from_input(url=None, image_path=None, html_path=None, text=None) -> Material:
        calls.append(("ingest", url, text))
        return Material(id="nowy-1", source=SourceKind.TEXT, url=url, text=text)

    ingest.material_from_input = material_from_input
    monkeypatch.setitem(sys.modules, "slad.ingest", ingest)

    export_pkg = types.ModuleType("slad.export")
    export_pkg.__path__ = []
    neo4j = types.ModuleType("slad.export.neo4j")
    graph = types.ModuleType("slad.export.graph")

    def export_neo4j(ws, out_dir):
        Path(out_dir, "nodes.csv").write_text("id\nm1\n", encoding="utf-8")
        Path(out_dir, "import.cypher").write_text("LOAD CSV ...", encoding="utf-8")

    def export_graphml(ws, path):
        Path(path).write_text("<graphml/>", encoding="utf-8")

    def export_json(ws, path):
        raise RuntimeError("awaria eksportu JSON")

    neo4j.export_neo4j = export_neo4j
    graph.export_graphml = export_graphml
    graph.export_json = export_json
    export_pkg.neo4j = neo4j
    export_pkg.graph = graph
    monkeypatch.setitem(sys.modules, "slad.export", export_pkg)
    monkeypatch.setitem(sys.modules, "slad.export.neo4j", neo4j)
    monkeypatch.setitem(sys.modules, "slad.export.graph", graph)

    eval_pkg = types.ModuleType("slad.eval")
    eval_pkg.__path__ = []
    run = types.ModuleType("slad.eval.run")

    def evaluate(ws, cfg):
        calls.append(("evaluate",))
        return {"pairwise": {"precision": 1.0, "recall": 0.5, "f1": 0.667}, "ari": 0.42}

    run.evaluate = evaluate
    eval_pkg.run = run
    monkeypatch.setitem(sys.modules, "slad.eval", eval_pkg)
    monkeypatch.setitem(sys.modules, "slad.eval.run", run)


PAGES = {
    "przeglad": "Przegląd",
    "kampanie": "Kampanie",
    "graf": "Graf powiązań",
    "material": "Materiał",
    "analiza": "Analiza nowego materiału",
    "metodyka": "Metodyka",
    "eksport": "Eksport",
}


def _goto(at, page: str):
    at.session_state["_goto"] = page
    return at.run()


def _no_errors(at) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_app_empty_workspace_shows_instructions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLAD_WORKSPACE", str(tmp_path / "pusty"))
    at = _apptest().run()
    _no_errors(at)
    assert at.title[0].value == "Przegląd"
    assert any("python -m slad demo" in i.value for i in at.info)
    for page, title in PAGES.items():
        _goto(at, page)
        _no_errors(at)
        assert at.title[0].value == title


def test_app_all_pages_render_on_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = build_workspace(tmp_path / "ws")
    run_info = {"dataset": "zbior.csv", "materials": 3, "time_s": 1.2, "finished_at": "2026-11-21T20:00:00+00:00"}
    ws.path("run.json").write_text(json.dumps(run_info), encoding="utf-8")
    calls: list = []
    _install_fakes(monkeypatch, calls)
    monkeypatch.setenv("SLAD_WORKSPACE", str(ws.root))
    at = _apptest().run()
    _no_errors(at)
    assert at.title[0].value == "Przegląd"
    assert [m.value for m in at.metric][:3] == ["3", "1", "2"]
    assert any("Ostatnie przetwarzanie" in e.label for e in at.expander)

    for page, title in PAGES.items():
        _goto(at, page)
        _no_errors(at)
        assert at.title[0].value == title, page

    # Kampanie: szczegóły wybranej kampanii, galeria z przyciskiem „Szczegóły” → strona materiału
    _goto(at, "kampanie")
    assert at.selectbox(key="campaign_pick").value == "K001"
    assert any("Dlaczego to jedna kampania?" in md.value for md in at.markdown)
    at.button(key="cg_K001_open_m2").click().run()
    _no_errors(at)
    assert at.title[0].value == "Materiał"
    assert at.selectbox(key="material_pick").value == "m2"

    # Graf: filtry i tryb „podążaj za śladem”
    _goto(at, "graf")
    at.toggle(key="g_hubs").set_value(True).run()
    _no_errors(at)
    at.selectbox(key="g_camp").set_value("K001").run()
    _no_errors(at)
    assert any("Widoczne: 2 materiałów" in c.value for c in at.caption)

    # Materiał m3 (bez obrazów i infrastruktury, z błędem) renderuje się bez wyjątków
    _goto(at, "material")
    at.selectbox(key="material_pick").set_value("m3").run()
    _no_errors(at)


def test_app_analysis_and_add_to_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = build_workspace(tmp_path / "ws")
    calls: list = []
    _install_fakes(monkeypatch, calls)
    monkeypatch.setenv("SLAD_WORKSPACE", str(ws.root))
    at = _apptest().run()
    _goto(at, "analiza")
    at.form_submit_button(key="an_submit").click().run()
    assert any("Podaj co najmniej jedno wejście" in w.value for w in at.warning)

    at.text_area(key="an_text").input("Jan Kowalski poleca inwestycję")
    at.checkbox(key="an_offline").check()
    at.form_submit_button(key="an_submit").click().run()
    _no_errors(at)
    assert ("analyze", "nowy-1", False, False) in calls
    assert any("Kampania: **K001" in md.value for md in at.markdown)
    assert any("pasuje do kampanii" in i.value for i in at.info)

    at.button(key="an_add").click().run()
    _no_errors(at)
    assert ("analyze", "nowy-1", True, False) in calls
    assert any("Materiał jest w bazie" in s.value for s in at.success)
    assert Workspace(ws.root).material("nowy-1") is not None


def test_app_export_and_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = build_workspace(tmp_path / "ws")
    calls: list = []
    _install_fakes(monkeypatch, calls)
    monkeypatch.setenv("SLAD_WORKSPACE", str(ws.root))
    at = _apptest().run()
    _goto(at, "eksport")
    _no_errors(at)
    captions = " ".join(c.value for c in at.caption)
    assert "awaria eksportu JSON" in captions  # błąd jednego eksportu nie psuje strony
    at.button(key="eval_run").click().run()
    _no_errors(at)
    assert ("evaluate",) in calls
    assert any(m.label == "ari" for m in at.metric)


def test_app_processing_panel_uses_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    _install_fakes(monkeypatch, calls)
    data = tmp_path / "zbior.csv"
    data.write_text("url\nhttps://a.example\n", encoding="utf-8")
    monkeypatch.setenv("SLAD_WORKSPACE", str(tmp_path / "ws"))
    at = _apptest().run()
    at.text_input(key="proc_path").set_value(str(data))
    at.button(key="proc_run").click().run()
    _no_errors(at)
    assert ("process", str(data)) in calls
    at.text_input(key="proc_path").set_value(str(tmp_path / "nie-ma")).run()
    at.button(key="proc_run").click().run()
    assert any("Nie znaleziono" in e.value for e in at.error)


def test_report_json_roundtrip() -> None:
    # raport ewaluacji może zawierać typy nie-JSON (np. datetime) — pobieranie używa default=str
    payload = json.dumps({"t": datetime(2026, 11, 21)}, default=str)
    assert "2026-11-21" in payload


def test_summary_and_legend_chips_escape_values() -> None:
    html = C.summary_html(
        [("👤 Osoby", [("<b>Jan</b>", 3)]), ("🏷️ Marki", []), ("🌐 Domeny", [(f"d{i}.pl", 1) for i in range(10)])]
    )
    assert "&lt;b&gt;Jan&lt;/b&gt;" in html and "<b>Jan" not in html
    assert "brak danych" in html and "… oraz 2 kolejnych" in html
    chips = G.legend_chips_html(
        [
            {"type": "campaign", "color": "#2a78d6", "shape": "square", "label": "K1 <i>x</i>"},
            {"type": "relation", "color": "#52514e", "dashes": True, "label": "powiązany (2)"},
            {"type": "campaign", "color": "expression(alert(1))", "label": "zły kolor"},
        ]
    )
    assert "■" in chips and "dashed" in chips and "&lt;i&gt;" in chips and "expression(" not in chips


def test_preferred_workspace(tmp_path: Path) -> None:
    default, demo, other = tmp_path / "default", tmp_path / "demo", tmp_path / "inny"
    for d in (default, demo, other):
        d.mkdir()
    (demo / "materials.jsonl").write_text("", encoding="utf-8")
    assert C.preferred_workspace([default, demo, other]) == demo  # pusty domyślny → katalog z danymi
    assert C.preferred_workspace([default, demo], explicit=True) == default  # SLAD_WORKSPACE wygrywa
    (default / "materials.jsonl").write_text("", encoding="utf-8")
    assert C.preferred_workspace([default, demo]) == default
    assert C.preferred_workspace([]) is None
