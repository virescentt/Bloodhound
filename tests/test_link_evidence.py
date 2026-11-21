"""Powiązania: wskaźniki, rzadkość wartości, porównanie par (przesłanki, wynik, relacja)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest
from datasketch import MinHash

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.link.evidence import LinkContext, compare, score_pair  # noqa: E402
from slad.link.indicators import indicators, page_is_generic, pretty_tracker  # noqa: E402
from slad.link.stats import corpus_stats, rarity_factor, rarity_from_df  # noqa: E402
from slad.models import (  # noqa: E402
    CorpusStats,
    Face,
    FeatureSet,
    ImageFeatures,
    InfraFeatures,
    Relation,
    TextFeatures,
)
from slad.settings import Settings  # noqa: E402

# ---------------------------------------------------------------- pomocnicze


_PERMS = MinHash(num_perm=128, seed=1, scheme="affine32").permutations


def minhash(text: str) -> list[int]:
    """Sygnatura jak w slad.features.text (szyngle 3 słów, datasketch affine32, seed 1)."""
    words = text.lower().split()
    if len(words) < 3:
        return []
    m = MinHash(num_perm=128, seed=1, scheme="affine32", permutations=_PERMS)
    m.update_batch({" ".join(words[i : i + 3]).encode() for i in range(len(words) - 2)})
    return [int(v) for v in m.hashvalues]


def image(phash: str, sha: str | None = None, faces: list[Face] | None = None, dhash: str = "0f0f0f0f0f0f0f0f"):
    return ImageFeatures(
        path=f"/tmp/{phash}.png",
        sha256=sha or ("s" + phash),
        width=600,
        height=400,
        phash=phash,
        dhash=dhash,
        whash=phash,
        colorhash="07000000000",
        faces=faces or [],
    )


def face(vec: list[float], identity: str | None = None) -> Face:
    return Face(bbox=(0, 0, 10, 10), score=0.9, embedding=vec, identity=identity)


def make(
    mid: str,
    text: str = "",
    *,
    persons=(),
    brands=(),
    narratives=(),
    keywords=(),
    iocs: dict | None = None,
    images=(),
    **infra,
) -> FeatureSet:
    tf = TextFeatures(
        input_text=text,
        text=text,
        persons=list(persons),
        brands=list(brands),
        narratives=list(narratives),
        keywords=list(keywords),
        iocs=iocs or {},
        minhash=minhash(text),
    )
    fs = FeatureSet(material_id=mid, text=tf, images=list(images), infra=InfraFeatures(**infra))
    av = []
    if text:
        av.append("tekst")
    if images:
        av.append("obraz")
    if any(img.faces for img in images):
        av.append("twarze")
    if fs.infra.domains or fs.infra.tracker_ids or fs.infra.ips or fs.infra.favicon_mmh3 is not None:
        av.append("infrastruktura")
    fs.available = av
    return fs


def ctx_for(*features: FeatureSet, cfg: Settings | None = None, filler: int = 0) -> LinkContext:
    """Kontekst porównań; `filler` dodaje niepowiązane materiały (realistyczne statystyki rzadkości)."""
    extra = [make(f"filler{i:03d}", f"zupełnie inny tekst numer {i} o pogodzie i ogrodzie") for i in range(filler)]
    return LinkContext(cfg or Settings(), [*features, *extra])


AD = (
    "Marek Zawadzki ujawnił sekret: platforma Zyskomat Pro daje 5000 zł dziennie bez ryzyka. "
    "Zarejestruj się dziś, liczba miejsc jest ograniczona, konsultant zadzwoni w ciągu godziny."
)
AD_VARIANT = (
    "Marek Zawadzki ujawnił sekret! Platforma Zyskomat Pro daje 4800 zł dziennie bez ryzyka. "
    "Zarejestruj się teraz, liczba miejsc jest ograniczona, konsultant zadzwoni w ciągu godziny."
)
OTHER = (
    "Rządowy program dopłat: każdy emeryt otrzyma 3000 zł jednorazowego wsparcia. Wypełnij wniosek "
    "online i podaj dane karty, aby odebrać środki jeszcze w tym miesiącu."
)


# ---------------------------------------------------------------- wskaźniki


def test_indicators_normalize_and_stoplist():
    fs = make(
        "a",
        iocs={
            "phone": ["+48 512 345 678"],
            "email": ["Kontakt@Zysk24.test"],
            "telegram": ["https://t.me/Doradca_Zysk"],
            "crypto_eth": ["0xABCDEF0123456789abcdef0123456789ABCDEF01"],
            "domain": ["www.onet.pl", "promo.zysk24.test"],
        },
        domains=["zysk24.test", "facebook.com"],
        hosts=["promo.zysk24.test", "l.facebook.com"],
        input_urls=["https://l.facebook.com/l.php?u=x", "https://bit.ly/3AbCdEf"],
        final_url="https://promo.zysk24.test/lp",
        tracker_ids=["FBPIXEL:998877665544", "ga:UA-XXXXX-Y"],
        ips=["203.0.113.7", "10.0.0.1"],
        asns=["AS64500 EXAMPLE-NET", "AS13335 CLOUDFLARENET"],
        nameservers=["anna.ns.cloudflare.com", "ns1.operator-dns.test"],
        title="Zyskomat Pro – oficjalna platforma",
        favicon_mmh3=-12345,
    )
    ind = indicators(fs)
    assert ind["phone"] == {"+48512345678"}
    assert ind["email"] == {"kontakt@zysk24.test"}
    assert ind["telegram"] == {"doradca_zysk"}
    assert ind["crypto_wallet"] == {"0xabcdef0123456789abcdef0123456789abcdef01"}
    assert ind["domain"] == {"zysk24.test"}  # onet.pl, facebook.com, bit.ly — wykluczone
    assert ind["host"] == {"promo.zysk24.test"}
    assert ind["short_url"] == {"bit.ly/3AbCdEf"}  # ścieżka skrótu z zachowaniem wielkości liter
    assert ind["tracker_id"] == {"fbpixel:998877665544"}  # placeholder UA-XXXXX-Y odrzucony
    assert ind["ip"] == {"203.0.113.7"}  # adres prywatny odrzucony
    assert ind["asn"] == {"as64500"}  # Cloudflare na liście wykluczeń
    assert ind["nameserver"] == {"ns1.operator-dns.test"}
    assert ind["title"] == {"zyskomat pro – oficjalna platforma"}
    assert ind["favicon"] == {"-12345"}
    assert pretty_tracker("fbpixel:998877665544") == "Facebook Pixel 998877665544"
    assert pretty_tracker("gtm:gtm-ab12") == "Google Tag Manager GTM-AB12"


def test_generic_page_features_are_dropped():
    challenge = make(
        "c", domains=["zysk24.test"], final_url="https://zysk24.test/", title="Just a moment...", favicon_mmh3=777
    )
    assert page_is_generic(challenge)
    ind = indicators(challenge)
    assert "title" not in ind and "favicon" not in ind
    assert ind["domain"] == {"zysk24.test"}
    on_platform = make(
        "p",
        input_urls=["https://scam-landing.test/x"],
        redirect_chain=["https://scam-landing.test/x", "https://www.facebook.com/login"],
        final_url="https://www.facebook.com/login",
        title="Facebook – log in or sign up",
        tracker_ids=["fbpixel:111222333444"],
        ips=["157.240.1.35"],
    )
    ind = indicators(on_platform)
    assert ind.get("domain") == {"scam-landing.test"}  # cloaker odsyła na platformę → domena oszustwa zostaje
    assert "tracker_id" not in ind and "ip" not in ind and "title" not in ind


def test_redirect_domains_and_cdn_ips():
    fs = make(
        "r",
        input_urls=["https://krotko.example/a1b2c3"],
        redirect_chain=["https://krotko.example/a1b2c3", "https://landing.test/oferta"],
        final_url="https://landing.test/oferta",
        ips=["198.51.100.10"],
        asns=["AS13335 CLOUDFLARENET"],
    )
    ind = indicators(fs)
    assert ind["domain"] == {"landing.test"}
    assert ind["redirect_domain"] == {"krotko.example"}
    assert "ip" not in ind  # materiał wyłącznie za CDN — adresy brzegowe nie są śladem


def test_persons_merge_text_names_and_face_identities():
    fs = make("f", persons=["Łukasz Nowak"], images=[image("f0f0f0f0f0f0f0f0", faces=[face([1, 0], "Lukasz_Nowak")])])
    assert indicators(fs)["person"] == {"lukasz nowak"}


# ---------------------------------------------------------------- rzadkość


def test_rarity_factor_decays_for_common_values():
    assert rarity_from_df(1, 100) == 1.0
    assert rarity_from_df(2, 100) == 1.0
    assert rarity_from_df(10, 100) == 1.0  # wartość jednej (dużej) kampanii nie jest karana
    common = rarity_from_df(60, 100)
    assert 0.05 <= common < 0.25
    assert rarity_from_df(100, 100) < common
    stats = CorpusStats(n=100, df={"asn": {"as64500": 70}})
    assert rarity_factor("asn", "as64500", stats) < 0.2
    assert rarity_factor("asn", "as64500", stats, enabled=False) == 1.0
    assert rarity_factor("asn", "unknown", stats) == 1.0


def test_corpus_stats_keeps_only_shared_values():
    feats = [make(f"m{i}", nameservers=["ns1.wspolny-dns.test"], domains=[f"d{i}.test"]) for i in range(5)]
    stats = corpus_stats(feats)
    assert stats.n == 5
    assert stats.df["nameserver"] == {"ns1.wspolny-dns.test": 5}
    assert "domain" not in stats.df  # każda domena tylko raz


# ---------------------------------------------------------------- porównanie par


def test_shared_tracker_gives_link_with_evidence():
    a = make("a", AD, tracker_ids=["fbpixel:998877665544"], domains=["alfa.test"])
    b = make("b", OTHER, tracker_ids=["fbpixel:998877665544"], domains=["beta.test"])
    link = compare(a, b, ctx_for(a, b, filler=10))
    assert link is not None
    assert link.relation == Relation.RELATED
    top = link.evidence[0]
    assert top.kind == "tracker_id" and top.group == "infrastruktura"
    assert "Facebook Pixel 998877665544" in top.detail
    assert link.score == pytest.approx(sum(e.weight for e in link.evidence), abs=1e-6)


def test_shared_cloudflare_asn_alone_gives_no_link():
    a = make("a", AD, asns=["AS13335 CLOUDFLARENET"], ips=["104.16.1.1"], domains=["alfa.test"])
    b = make("b", OTHER, asns=["AS13335 CLOUDFLARENET"], ips=["104.16.1.1"], domains=["beta.test"])
    assert compare(a, b, ctx_for(a, b, filler=10)) is None
    # nawet bez listy wykluczeń wspólny ASN to przesłanka miękka o wadze 0.5
    c = make("c", OTHER, asns=["AS64500 HOSTING"], domains=["gamma.test"])
    d = make("d", OTHER + " Dodatkowe zdanie.", asns=["AS64500 HOSTING"], domains=["delta.test"])
    assert compare(c, d, ctx_for(c, d, filler=10)) is None or compare(c, d, ctx_for(c, d)).evidence[0].kind != "asn"


def test_same_celebrity_alone_gives_no_link():
    emb = [0.6, 0.8, 0.0]
    a = make(
        "a",
        AD,
        persons=["Marek Zawadzki"],
        narratives=["inwestycja_krypto"],
        brands=["Info24"],
        keywords=["bitcoin", "zysk", "platforma"],
        images=[image("8f3a5c7e9b1d2f40", faces=[face(emb, "Marek_Zawadzki")])],
    )
    b = make(
        "b",
        OTHER,
        persons=["Marek Zawadzki"],
        narratives=["inwestycja_krypto"],
        brands=["Info24"],
        keywords=["bitcoin", "zysk", "platforma"],
        images=[image("71c5a3816e4bd0bf", faces=[face(emb, "Marek_Zawadzki")])],
    )
    ctx = ctx_for(a, b, filler=10)
    assert compare(a, b, ctx) is None
    ps = score_pair(a, b, ctx)
    kinds = {e.kind for e in ps.evidence}
    assert {"person", "face", "narrative"} <= kinds  # przesłanki widoczne w wyjaśnieniu…
    assert ps.score < 4.0  # …ale za słabe, by powiązać


def test_near_duplicate_text_and_image_give_variant_or_duplicate():
    a = make("a", AD, images=[image("8f3a5c7e9b1d2f40")])
    b = make("b", AD_VARIANT, images=[image("8f3a5c7e9b1d2f43")])  # pHash Δ=2
    link = compare(a, b, ctx_for(a, b, filler=10))
    assert link is not None
    assert link.relation in (Relation.VARIANT, Relation.DUPLICATE)
    kinds = {e.kind for e in link.evidence}
    assert "image_phash" in kinds and ("text_minhash" in kinds or "text_tfidf" in kinds)
    assert any("pHash Δ=2" in e.detail for e in link.evidence)


def test_exact_duplicate_is_labelled_duplicate():
    a = make("a", AD, images=[image("8f3a5c7e9b1d2f40", sha="abc")])
    b = make("b", AD, images=[image("8f3a5c7e9b1d2f40", sha="abc")])
    link = compare(a, b, ctx_for(a, b, filler=10))
    assert link is not None and link.relation == Relation.DUPLICATE
    assert any(e.detail.startswith("identyczny plik graficzny") for e in link.evidence)


def test_same_landing_different_creatives_is_not_duplicate():
    infra = dict(
        domains=["zysk24.test"],
        final_url="https://zysk24.test/lp",
        tracker_ids=["gtm:GTM-AB12CD", "fbpixel:998877665544"],
        favicon_mmh3=4242,
        title="Zyskomat Pro – oficjalna platforma",
    )
    a = make("a", AD, images=[image("8f3a5c7e9b1d2f40")], **infra)
    b = make("b", OTHER, images=[image("71c5a3816e4bd0bf")], **infra)
    link = compare(a, b, ctx_for(a, b, filler=10))
    assert link is not None
    assert link.score >= 12.0
    assert link.relation == Relation.VARIANT  # wspólna strona docelowa, ale różne kreacje


def test_missing_groups_compare_on_available_features():
    a = make("a", AD)  # sam tekst
    b = make("b", AD_VARIANT, domains=["zysk24.test"], images=[image("8f3a5c7e9b1d2f40")])  # tekst + obraz + infra
    link = compare(a, b, ctx_for(a, b, filler=10))
    assert link is not None
    assert {e.group for e in link.evidence} == {"tekst"}
    assert 0 < link.coverage < 1
    empty = make("e")
    assert compare(a, empty, ctx_for(a, empty)) is None


def test_single_medium_indicator_needs_corroboration():
    # ta sama (np. domyślna) favicona i ta sama narracja — za mało; favicona + tytuł strony — wystarczy
    a = make("a", AD, narratives=["inwestycja_krypto"], favicon_mmh3=4242, domains=["alfa.test"])
    b = make("b", OTHER, narratives=["inwestycja_krypto"], favicon_mmh3=4242, domains=["beta.test"])
    assert compare(a, b, ctx_for(a, b, filler=10)) is None
    c = make("c", AD, favicon_mmh3=4242, title="Zyskomat Pro – oficjalna platforma", domains=["alfa.test"])
    d = make("d", OTHER, favicon_mmh3=4242, title="Zyskomat Pro – oficjalna platforma", domains=["beta.test"])
    assert compare(c, d, ctx_for(c, d, filler=10)) is not None


def test_registrar_and_creation_date_is_soft_evidence():
    kw = dict(registrar="NameCheap, Inc.")
    a = make("a", AD, domains=["alfa.test"], domain_created=datetime(2026, 11, 1, 10), **kw)
    b = make("b", OTHER, domains=["beta.test"], domain_created=datetime(2026, 11, 2, 9), **kw)
    ps = score_pair(a, b, ctx_for(a, b, filler=10))
    ev = [e for e in ps.evidence if e.kind == "registrar_created"]
    assert ev and "NameCheap" in ev[0].detail
    assert ps.relation is None


def test_rarity_weakens_values_shared_by_many_materials():
    shared = dict(ips=["203.0.113.50"], favicon_mmh3=99, asns=["AS64501 HOSTING"])
    topics = [
        "pogoda",
        "ogród",
        "kuchnia",
        "rower",
        "góry",
        "morze",
        "kino",
        "muzyka",
        "szachy",
        "piłka",
        "teatr",
        "las",
    ]
    crowd = [make(f"h{i}", f"{t} " * 3 + f"tekst {i * 7919}", **shared) for i, t in enumerate(topics)]
    ctx = LinkContext(Settings(), crowd)
    ps = score_pair(crowd[0], crowd[1], ctx)
    assert ps.relation is None
    ip = [e for e in ps.evidence if e.kind == "ip"]
    assert ip and ip[0].weight < 2.5 and "waga obniżona" in ip[0].detail


def test_compare_never_crashes_on_broken_features():
    a = make("a", AD, images=[image("not-a-hash")])
    a.text.minhash = [1, 2, 3]  # zbyt krótka sygnatura
    a.images[0].faces = [face([float("nan"), 1.0])]
    b = make("b", AD, images=[image("zzzz")])
    ctx = ctx_for(a, b)
    link = compare(a, b, ctx)  # tekst identyczny — powiązanie mimo uszkodzonych cech obrazu
    assert link is not None and link.relation == Relation.DUPLICATE


def test_context_add_new_material_uses_fitted_index():
    a = make("a", AD, domains=["alfa.test"])
    b = make("b", OTHER, domains=["beta.test"])
    ctx = ctx_for(a, b, filler=5)
    new = make("new", AD_VARIANT)
    p = ctx.add(new)
    assert not ctx.is_base(p)
    assert ctx.tfidf_sim("new", "a") > 0.8 > ctx.tfidf_sim("new", "b")
    assert compare(new, a, ctx) is not None


# ---------------------------------------------------------------- odporność na „tło” i rzadkie ślady


def _flip(hex_hash: str, *bits: int) -> str:
    v = int(hex_hash, 16)
    for b in bits:
        v ^= 1 << b
    return f"{v:016x}"


def test_generic_layout_image_is_weakened_by_neighbourhood_density():
    layout = "8f3a5c7e9b1d2f40"  # ten sam układ zrzutu ekranu platformy u wielu niezależnych materiałów
    crowd = [
        make(
            f"u{i}", f"tekst numer {i} " + "inne słowa " * (i % 5), images=[image(_flip(layout, i % 64, (i * 7) % 64))]
        )
        for i in range(30)
    ]
    ctx = LinkContext(Settings(), crowd)
    ps = score_pair(crowd[0], crowd[1], ctx)
    img = [e for e in ps.evidence if e.kind == "image_phash"]
    assert img and img[0].weight < 2.0 and "waga obniżona" in img[0].detail
    assert ps.relation is None
    # ta sama para w zbiorze bez tłumu podobnych zrzutów — pełna waga
    alone = LinkContext(Settings(), crowd[:2] + [make(f"x{i}", f"inny {i}") for i in range(20)])
    img2 = [e for e in score_pair(crowd[0], crowd[1], alone).evidence if e.kind == "image_phash"]
    assert img2 and img2[0].weight > img[0].weight


def test_shared_tls_certificate_domains_link_materials():
    a = make("a", AD, domains=["alfa.test"], final_url="https://alfa.test/", tls_sans=["alfa.test", "beta.test"])
    b = make("b", OTHER, domains=["beta.test"], final_url="https://beta.test/", tls_sans=["beta.test"])
    link = compare(a, b, ctx_for(a, b, filler=10))
    assert link is not None
    assert link.evidence[0].kind == "cert_domain"
    assert "beta.test" in link.evidence[0].detail and "certyfikacie TLS" in link.evidence[0].detail


def test_dead_shortener_link_is_treated_as_redirect_not_landing():
    live = make(
        "live",
        AD,
        input_urls=["https://krotko.example/a1b2c3"],
        redirect_chain=["https://krotko.example/a1b2c3", "https://alfa.test/oferta"],
        final_url="https://alfa.test/oferta",
    )
    dead1 = make(
        "dead1", OTHER, input_urls=["https://krotko.example/ffee01"], redirect_chain=["https://krotko.example/ffee01"]
    )
    dead2 = make("dead2", "Zupełnie inna reklama kredytu bez BIK", input_urls=["https://krotko.example/0a0b0c"])
    ctx = ctx_for(live, dead1, dead2, filler=10)
    p = ctx.prepared(dead1)
    assert "krotko.example" in p.ind.get("redirect_domain", set())
    assert "krotko.example" not in p.ind.get("domain", set())
    assert compare(dead1, dead2, ctx) is None  # wspólny skracacz to nie ślad kampanii


def test_two_common_medium_signals_do_not_corroborate():
    # domyślna favicona hostingu i ten sam skracacz u wielu materiałów: dwie średnie, ale częste przesłanki
    shared = dict(favicon_mmh3=-1183975157)
    crowd = []
    for i in range(16):
        chain = [f"https://krotko.example/{i:06x}", f"https://site{i}.test/lp"]
        crowd.append(
            make(
                f"h{i}",
                f"zupełnie inna treść numer {i * 31} " + "słowo " * (i % 4),
                input_urls=chain[:1],
                redirect_chain=chain,
                final_url=chain[-1],
                **shared,
            )
        )
    ctx = LinkContext(Settings(), crowd + [make(f"z{i}", f"tło {i}") for i in range(10)])
    ps = score_pair(crowd[0], crowd[1], ctx)
    assert {e.kind for e in ps.evidence} >= {"favicon", "redirect_domain"}
    assert ps.relation is None


def test_layout_variant_is_variant_not_duplicate():
    infra = dict(domains=["zysk24.test"], final_url="https://zysk24.test/", tracker_ids=["fbpixel:998877665544"])
    a = make("a", AD, images=[image("8f3a5c7e9b1d2f40")], **infra)
    b = make("b", AD_VARIANT, images=[image(_flip("8f3a5c7e9b1d2f40", 3, 17, 40))], **infra)  # pHash Δ=3
    link = compare(a, b, ctx_for(a, b, filler=10))
    assert link is not None and link.relation == Relation.VARIANT


def test_tracker_software_fingerprints_and_offer_fields():
    fs = make("t", tracker_ids=["tracker:keitaro", "form:offer_id=4521", "gsv:AbCdEfGh123", "GTM-XXXX"])
    ind = indicators(fs)
    assert ind["tracker_id"] == {"gsv:abcdefgh123"}  # weryfikacja domeny = konto operatora
    assert ind["form_offer"] == {"offer_id=4521"}  # identyfikator oferty: przesłanka średnia


def test_short_identical_slogans_do_not_link():
    slogan = "Kliknij i sprawdź teraz! Oferta dnia"  # szablonowe hasło — ten sam tekst w niezależnych reklamach
    a = make("a", slogan, domains=["alfa.test"])
    b = make("b", slogan, domains=["beta.test"])
    ps = score_pair(a, b, ctx_for(a, b, filler=10))
    assert ps.relation is None
    assert any("krótki tekst" in e.detail for e in ps.evidence)
    long_a = make("c", AD, domains=["alfa.test"])
    long_b = make("d", AD, domains=["beta.test"])
    assert compare(long_a, long_b, ctx_for(long_a, long_b, filler=10)) is not None


def test_same_layout_with_different_ad_text_is_a_template_not_a_link():
    # ten sam układ kreacji (pHash Δ=2), ale zupełnie inna treść reklamy — wspólny szablon, nie ta sama kampania
    a = make("a", AD, images=[image("8f3a5c7e9b1d2f40")])
    b = make("b", OTHER, images=[image(_flip("8f3a5c7e9b1d2f40", 5, 33))])
    ps = score_pair(a, b, ctx_for(a, b, filler=10))
    img = [e for e in ps.evidence if e.kind == "image_phash"]
    assert img and "inny tekst reklamy" in img[0].detail
    assert ps.relation is None
    # ten sam układ i podobny tekst — wariant tej samej kreacji
    c = make("c", AD_VARIANT, images=[image(_flip("8f3a5c7e9b1d2f40", 5, 33))])
    assert compare(a, c, ctx_for(a, c, filler=10)) is not None


def test_moderate_text_similarity_does_not_corroborate_a_shared_template():
    # ten sam szkielet strony (zestaw „kit”) + teksty o tej samej narracji (cos ~0.5) — za mało
    tpl = dict(dom_simhash="0f0f0f0f0f0f0f0f")
    a = make("a", AD, narratives=["inwestycja_krypto"], domains=["alfa.test"], final_url="https://alfa.test/", **tpl)
    text_b = AD.replace("Marek Zawadzki ujawnił sekret", "Ekspert radzi").replace("Zyskomat Pro", "Kryptobot X")
    text_b = text_b.replace("konsultant zadzwoni w ciągu godziny", "doradca odpisze jeszcze dziś wieczorem")
    b = make(
        "b", text_b, narratives=["inwestycja_krypto"], domains=["beta.test"], final_url="https://beta.test/", **tpl
    )
    ctx = ctx_for(a, b, filler=10)
    ps = score_pair(a, b, ctx)
    kinds = {e.kind for e in ps.evidence}
    assert "dom_simhash" in kinds
    tfidf = [e for e in ps.evidence if e.kind == "text_tfidf"]
    assert tfidf and 0.35 <= tfidf[0].similarity < 0.7  # podobieństwo umiarkowane — nie potwierdza szablonu
    assert ps.relation is None


# ---------------------------------------------------------------- kanały treści (materiały częściowe)


def _channels(fs: FeatureSet, *, input_text: str = "", ocr_text: str = "", page_text: str = "") -> FeatureSet:
    """Materiał z tekstem rozdzielonym na kanały (wejście / OCR / strona) — jak z slad.features.text."""
    joined = "\n".join(p for p in (input_text, ocr_text, page_text) if p)
    fs.text.input_text, fs.text.ocr_text, fs.text.page_text = input_text, ocr_text, page_text
    fs.text.text = joined
    fs.text.minhash = minhash(joined)
    return fs


PAGE = (
    "Witamy na platformie inwestycyjnej. Zarejestruj konto, wpłać depozyt startowy i obserwuj, jak algorytm "
    "pomnaża Twoje środki. Regulamin, polityka prywatności, kontakt z działem obsługi klienta. " * 3
)


def test_same_ad_text_as_text_field_and_as_screenshot_ocr_links():
    # pełny rekord: tekst posta + napis na kreacji (OCR) + strona docelowa; nowy materiał: sam zrzut ekranu,
    # którego OCR zawiera ten sam tekst reklamy — złączone teksty różnią się, kanały są niemal identyczne
    full = _channels(make("full"), input_text=AD, ocr_text="PILNE! Sekret Marka Zawadzkiego", page_text=PAGE)
    shot = _channels(make("shot"), ocr_text=AD_VARIANT.replace("ujawnił", "ujawnil"))
    ctx = ctx_for(full, shot, filler=10)
    lk = compare(full, shot, ctx)
    assert lk is not None
    text = [e for e in lk.evidence if e.kind in ("text_minhash", "text_tfidf")]
    assert text and any("OCR" in e.detail for e in text)
    # niezależna reklama na zrzucie ekranu — kanały nie są podobne, brak powiązania
    other = _channels(make("other"), ocr_text=OTHER)
    assert compare(full, other, ctx_for(full, other, filler=10)) is None


def test_text_only_variant_with_typo_links_to_full_record():
    # ten sam tekst reklamy z literówką (zaciemnianie / OCR) jako sam tekst vs pełny rekord ze zrzutem i stroną
    full = _channels(make("full"), input_text=AD, ocr_text="Zyskomat Pro — 5000 zł dziennie", page_text=PAGE)
    typo = _channels(make("typo"), input_text=AD.replace("ograniczona", "ogarniczona").replace("sekret", "sekert"))
    ctx = ctx_for(full, typo, filler=10)
    ps = score_pair(typo, full, ctx)
    assert ps.relation is not None
    mh = [e for e in ps.evidence if e.kind == "text_minhash"]
    assert mh and mh[0].similarity >= 0.8
    # nowy materiał spoza bazy (analiza) — ten sam wynik przez wektory liczone na żądanie
    ctx2 = ctx_for(full, filler=10)
    p_new = ctx2.add(typo)
    c, length, pair = ctx2.segment_sims(p_new, ctx2.prep["full"])
    assert c >= 0.9 and length >= 100 and pair == (0, 0)
    assert compare(typo, full, ctx2) is not None


def test_channel_similarity_sparse_path_matches_dense(monkeypatch):
    import slad.link.evidence as ev

    full = _channels(make("full"), input_text=AD, ocr_text="Zyskomat Pro — 5000 zł dziennie", page_text=PAGE)
    shot = _channels(make("shot"), ocr_text=AD_VARIANT)
    dense = ctx_for(full, shot, filler=5)
    expected = dense.segment_sims(dense.prep["full"], dense.prep["shot"])
    monkeypatch.setattr(ev, "DENSE_MAX", 0)  # duże zbiory: bez macierzy gęstej, iloczyny wierszy CSR
    sparse = ctx_for(full, shot, filler=5)
    assert sparse.Sseg is None
    got = sparse.segment_sims(sparse.prep["full"], sparse.prep["shot"])
    assert got[1:] == expected[1:] and got[0] == pytest.approx(expected[0], abs=1e-5)
    assert expected[2] == (0, 1)  # wejście pełnego rekordu ↔ OCR zrzutu


def test_identical_blank_placeholder_image_does_not_link():
    # ten sam plik zaślepki (pusty / jednolity obraz: bez pHash) w dwóch niezależnych materiałach — brak przesłanki
    blank = ImageFeatures(
        path="/tmp/blank.png", sha256="same-placeholder", width=600, height=400, phash="", dhash="", whash="",
        colorhash="",
    )  # fmt: skip
    a = make("a", AD, images=[blank])
    b = make("b", OTHER, images=[blank.model_copy()])
    ps = score_pair(a, b, ctx_for(a, b, filler=10))
    assert not [e for e in ps.evidence if e.kind == "image_phash"]
    assert ps.relation is None
    # ten sam plik z treścią (pHash obecny) nadal jest przesłanką „identyczny plik graficzny”
    c = make("c", OTHER, images=[image("8f3a5c7e9b1d2f40", sha="same-creative")])
    d = make("d", OTHER, images=[image("8f3a5c7e9b1d2f40", sha="same-creative")])
    img = [e for e in score_pair(c, d, ctx_for(c, d, filler=10)).evidence if e.kind == "image_phash"]
    assert img and "identyczny plik" in img[0].detail


def test_scam_site_on_free_website_builder_keeps_its_own_identity():
    # strony użytkowników kreatorów (wordpress.com, tilda.ws) są osobnymi witrynami, choć sama platforma jest wyjątkiem
    a = make("a", AD, final_url="https://zysk-teraz.wordpress.com/start", domains=["zysk-teraz.wordpress.com"])
    b = make("b", OTHER, final_url="https://zysk-teraz.wordpress.com/rejestracja", domains=["zysk-teraz.wordpress.com"])
    c = make("c", OTHER, final_url="https://inna-strona.wordpress.com/", domains=["inna-strona.wordpress.com"])
    assert "zysk-teraz.wordpress.com" in indicators(a).get("domain", set())
    assert not page_is_generic(a)  # strona na kreatorze to nie „platforma” — jej cechy (trackery, IP…) się liczą
    ctx = ctx_for(a, b, c, filler=10)
    lk = compare(a, b, ctx)
    assert lk is not None and any(e.kind == "domain" for e in lk.evidence)
    assert compare(a, c, ctx) is None  # dwie różne witryny na tym samym kreatorze — brak powiązania
