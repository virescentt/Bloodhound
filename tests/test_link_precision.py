"""Precyzja powiązań: wspólny szablon tekstu i certyfikaty współdzielone nie wiążą niezależnych materiałów."""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import pytest
from datasketch import MinHash

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.link import build  # noqa: E402
from slad.link.evidence import LinkContext, compare, score_pair  # noqa: E402
from slad.link.indicators import indicators  # noqa: E402
from slad.models import FeatureSet, InfraFeatures, Relation, TextFeatures  # noqa: E402
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


def make(mid: str, text: str = "", *, iocs: dict | None = None, **infra) -> FeatureSet:
    tf = TextFeatures(input_text=text, text=text, iocs=iocs or {}, minhash=minhash(text))
    fs = FeatureSet(material_id=mid, text=tf, infra=InfraFeatures(**infra))
    fs.available = ["tekst"] if text else []
    if fs.infra.domains or fs.infra.favicon_mmh3 is not None:
        fs.available.append("infrastruktura")
    return fs


def filler(n: int) -> list[FeatureSet]:
    return [
        make(f"z{i:03d}", f"zupełnie inny tekst numer {i} o pogodzie, ogrodzie i kuchni {i * 31}") for i in range(n)
    ]


# ogólnikowy tekst reklamy „inwestycyjnej” — ten sam szablon u wielu niezależnych reklamodawców
GENERIC = (
    "Nowa platforma inwestycyjna pozwala zarabiać nawet kilka tysięcy złotych tygodniowo bez wychodzenia z domu. "
    "Wystarczy zarejestrować konto, wpłacić niewielki depozyt startowy i pozwolić, aby inteligentny algorytm "
    "handlował za ciebie na rynkach walutowych i kryptowalutowych. System działa całą dobę, a zyski trafiają "
    "bezpośrednio na twoje konto bankowe. Liczba miejsc w programie jest ograniczona, dlatego nie zwlekaj "
    "i zapisz się jeszcze dziś, zanim oferta zostanie zamknięta."
)
AD = (
    "Marek Zawadzki ujawnił sekret: platforma Zyskomat Pro daje 5000 zł dziennie bez ryzyka. "
    "Zarejestruj się dziś, liczba miejsc jest ograniczona, konsultant zadzwoni w ciągu godziny."
)
OTHER = (
    "Rządowy program dopłat: każdy emeryt otrzyma 3000 zł jednorazowego wsparcia. Wypełnij wniosek "
    "online i podaj dane karty, aby odebrać środki jeszcze w tym miesiącu."
)


def generic(i: int, unique: int = 3) -> str:
    """Szablon GENERIC z `unique` słowami unikalnymi dla materiału `i` wstawionymi w różne miejsca."""
    words = GENERIC.split()
    for k in range(unique):
        words.insert((i * 7 + k * 13) % len(words), f"slowo{i}x{k}")
    return " ".join(words)


# ---------------------------------------------------------------- wspólny szablon tekstu


@pytest.mark.parametrize("unique", [3, 0])
def test_generic_text_shared_by_many_unrelated_materials_does_not_link(unique):
    feats = [make(f"g{i:02d}", generic(i, unique)) for i in range(30)]
    ctx = LinkContext(Settings(), feats)
    scores = [score_pair(a, b, ctx) for a, b in combinations(feats, 2)]
    assert all(ps.relation is None and ps.score < 4.0 for ps in scores)  # 30 kopii tego samego tekstu to „tło”
    assert not any(ps.relation == Relation.DUPLICATE for ps in scores)
    text = [e for e in scores[0].evidence if e.kind in ("text_minhash", "text_tfidf")]
    assert text and all("szablon" in e.detail and "waga obniżona" in e.detail for e in text)  # widoczne w wyjaśnieniu
    links, campaigns, _ = build(feats, Settings())
    assert links == [] and campaigns == []


def test_template_text_above_fraction_needs_another_channel():
    # 9 z 40 materiałów (22%) z tym samym tekstem: waga bez kary, ale sam tekst nie wystarcza do powiązania…
    band = [make(f"b{i}", generic(i)) for i in range(9)]
    ctx = LinkContext(Settings(), band + filler(31))
    for a, b in combinations(band, 2):
        ps = score_pair(a, b, ctx)
        assert ps.relation is None and ps.score < 4.0
    # …a ten sam tekst potwierdzony inną przesłanką (wspólna favicona strony) wiąże jak dotąd
    infra = dict(favicon_mmh3=4242)
    pair = [make(f"b{i}", generic(i), domains=[f"strona{i}.test"], **infra) for i in range(2)]
    ctx = LinkContext(Settings(), pair + band[2:] + filler(31))
    lk = compare(pair[0], pair[1], ctx)
    assert lk is not None and {"favicon", "text_minhash"} <= {e.kind for e in lk.evidence}


def test_campaign_variants_sharing_template_text_and_identifier_still_link():
    background = [make(f"g{i:02d}", generic(i)) for i in range(30)]
    by_phone = [make(f"p{i}", generic(100 + i), iocs={"phone": ["+48 512 345 678"]}) for i in range(5)]
    by_domain = [
        make(f"d{i}", generic(200 + i), domains=["zysk24.test"], final_url="https://zysk24.test/lp") for i in range(5)
    ]
    feats = background + by_phone + by_domain
    ctx = LinkContext(Settings(), feats)
    for camp in (by_phone, by_domain):
        for a, b in combinations(camp, 2):
            lk = compare(a, b, ctx)
            assert lk is not None and lk.relation != Relation.DUPLICATE  # tekst identyczny, ale wspólny z „tłem”
            assert lk.evidence[0].kind in ("phone", "domain")
    links, campaigns, _ = build(feats, Settings())
    assert [c.members for c in campaigns] == [[f"d{i}" for i in range(5)], [f"p{i}" for i in range(5)]]
    assert all(lk.a[0] == lk.b[0] and lk.a[0] in "pd" for lk in links)


def test_identical_rare_text_alone_is_not_a_duplicate():
    a, b = make("a", AD), make("b", AD)
    lk = compare(a, b, LinkContext(Settings(), [a, b, *filler(10)]))
    assert lk is not None and lk.relation != Relation.DUPLICATE  # rzadki tekst wiąże, ale bez innego kanału
    c, d = make("c", AD, domains=["zysk24.test"]), make("d", AD, domains=["zysk24.test"])
    lk = compare(c, d, LinkContext(Settings(), [c, d, *filler(10)]))
    assert lk is not None and lk.relation == Relation.DUPLICATE  # ten sam tekst i ta sama domena


# ---------------------------------------------------------------- certyfikaty współdzielone


def test_shared_hosting_certificate_gives_no_tls_evidence():
    sans = [f"klient{i}.test" for i in range(40)]  # certyfikat hostingu: 40 domen różnych klientów
    cert = dict(tls_sans=sans, tls_fingerprint="ab" * 32)
    a = make("a", AD, domains=["klient0.test"], final_url="https://klient0.test/", **cert)
    b = make("b", OTHER, domains=["klient1.test"], final_url="https://klient1.test/", **cert)
    for fs in (a, b):
        ind = indicators(fs)
        assert "tls_fingerprint" not in ind and "cert_domain" not in ind
    ps = score_pair(a, b, LinkContext(Settings(), [a, b, *filler(10)]))
    assert not {"tls_fingerprint", "cert_domain"} & {e.kind for e in ps.evidence}
    assert ps.relation is None
    # SAN wieloznaczny i długa lista domen z CT — tak samo
    wildcard = make("w", AD, tls_sans=["*.hosting.test", "hosting.test"], tls_fingerprint="cd" * 32)
    assert not {"tls_fingerprint", "cert_domain"} & set(indicators(wildcard))
    assert "cert_domain" not in indicators(make("ct", AD, ct_domains=sans))
    # certyfikat operatora na kilka własnych domen nadal jest przesłanką
    own = make("o", AD, final_url="https://alfa.test/", tls_sans=["alfa.test", "beta.test"], tls_fingerprint="ef" * 32)
    ind = indicators(own)
    assert ind["tls_fingerprint"] == {"ef" * 32} and ind["cert_domain"] == {"alfa.test", "beta.test"}
