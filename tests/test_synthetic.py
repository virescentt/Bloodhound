"""Zbiór syntetyczny: kompletność, determinizm, zarezerwowane przestrzenie nazw i trudne przypadki."""

from __future__ import annotations

import hashlib
import ipaddress
import itertools
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:  # uruchomienie `pytest` bez instalacji pakietu i bez wspólnego conftest.py
    sys.path.insert(0, _ROOT)

from slad.ingest import load_dataset  # noqa: E402
from slad.ingest.synthetic import PERSONAS, generate  # noqa: E402
from slad.models import InfraFeatures, SourceKind  # noqa: E402

FIELDS = {"id", "url", "text", "image", "html", "platform", "advertiser", "date", "campaign", "infra"}
TEST_NETS = [ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("syn")
    t0 = time.process_time()  # czas CPU — odporny na obciążenie maszyny
    path = generate(out)
    elapsed = time.process_time() - t0
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {"dir": out, "path": path, "records": records, "elapsed": elapsed}


def _images(r: dict[str, Any]) -> list[str]:
    img = r["image"]
    return [] if not img else ([img] if isinstance(img, str) else list(img))


def _domain(url: str) -> str:
    return re.sub(r"^https?://", "", url).split("/")[0].split("?")[0].split(":")[0].lower()


def _by_label(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        out[r["campaign"]].append(r)
    return out


def test_files_and_fields(dataset: dict[str, Any]) -> None:
    out, records = dataset["dir"], dataset["records"]
    assert dataset["path"] == out / "records.jsonl"
    assert (out / "README.md").read_text(encoding="utf-8").startswith("# Syntetyczny zbiór")
    assert 40 <= len(records) <= 150
    assert dataset["elapsed"] < 10  # cel: < 5 s CPU; luźny próg, żeby test nie był kruchy
    ids = [r["id"] for r in records]
    assert len(set(ids)) == len(ids)
    for r in records:
        assert set(r) == FIELDS
        for rel in _images(r):
            assert (out / rel).is_file()
        if r["html"]:
            assert (out / r["html"]).is_file()
        if r["infra"]:
            assert set(r["infra"]) <= set(InfraFeatures.model_fields)
            InfraFeatures.model_validate(r["infra"])  # zgodne z kontraktem


def test_labels_campaigns_and_noise(dataset: dict[str, Any]) -> None:
    groups = _by_label(dataset["records"])
    campaigns = [lab for lab in groups if lab.startswith("kampania-")]
    noise = [lab for lab in groups if lab.startswith("noise-")]
    assert len(campaigns) == 8 and len(noise) == 12
    assert sorted(noise) == [f"noise-{i:02d}" for i in range(1, 13)]
    assert all(len(groups[lab]) == 1 for lab in noise)  # szum: unikalne etykiety
    assert all(len(groups[lab]) >= 4 for lab in campaigns)


def test_deterministic_and_seed_dependent(tmp_path: Path) -> None:
    def snapshot(d: Path) -> tuple[str, dict[str, str]]:
        files = {
            p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*"))
            if p.is_file()
        }
        return (d / "records.jsonl").read_text(encoding="utf-8"), files

    a = snapshot(generate(tmp_path / "a", seed=3, n_campaigns=3, noise=2).parent)
    b = snapshot(generate(tmp_path / "b", seed=3, n_campaigns=3, noise=2).parent)
    c = snapshot(generate(tmp_path / "c", seed=4, n_campaigns=3, noise=2).parent)
    assert a == b
    assert a[0] != c[0]


def test_more_campaigns_than_base(tmp_path: Path) -> None:
    path = generate(tmp_path / "big", seed=1, n_campaigns=10, noise=14)
    groups = _by_label([json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()])
    assert sum(1 for lab in groups if lab.startswith("kampania-")) == 10
    assert sum(1 for lab in groups if lab.startswith("noise-")) == 14


def test_reserved_namespaces_only(dataset: dict[str, Any]) -> None:
    allowed_real = {"www.googletagmanager.com", "connect.facebook.net", "t.me"}  # skrypty trackerów i link Telegram
    for r in dataset["records"]:
        urls = [r["url"]] if r["url"] else []
        if r["infra"]:
            urls += r["infra"].get("redirect_chain", []) + [r["infra"].get("final_url") or ""]
            for ip in r["infra"].get("ips", []):
                assert any(ipaddress.ip_address(ip) in net for net in TEST_NETS), ip
            for ns in r["infra"].get("nameservers", []):
                assert ns.endswith((".test", ".example"))
        for u in filter(None, urls):
            assert _domain(u).endswith((".test", ".example")), u
    for page in (dataset["dir"] / "pages").glob("*.html"):
        for u in re.findall(r'(?:href|src|action)="(https?://[^"]+)"', page.read_text(encoding="utf-8")):
            d = _domain(u)
            assert d.endswith((".test", ".example")) or d in allowed_real, (page.name, u)


def test_campaign_signals_shared_within_campaign(dataset: dict[str, Any]) -> None:
    out = dataset["dir"]
    for label, recs in _by_label(dataset["records"]).items():
        if not label.startswith("kampania-"):
            continue
        pages = [(out / r["html"]).read_text(encoding="utf-8") for r in recs if r["html"]]
        forms = {m for p in pages for m in re.findall(r'<form[^>]+action="([^"]+)"', p)}
        assert len(forms) == 1, label  # wspólny endpoint formularza leadowego
        gas = [set(re.findall(r"gtag\('config','(G-[A-Z0-9]+)'\)", p)) for p in pages]
        if any(gas):
            assert len(set.union(*gas)) == 1, label  # ten sam identyfikator GA4 na wszystkich domenach
        favicons = {r["infra"]["favicon_mmh3"] for r in recs if r["infra"] and "favicon_mmh3" in r["infra"]}
        assert len(favicons) <= 1, label
        domains = {_domain(r["infra"]["final_url"]) for r in recs if r["infra"] and r["infra"].get("final_url")}
        assert 1 <= len(domains) <= 4, label  # rotacja 2–4 domen


def test_hard_cases(dataset: dict[str, Any]) -> None:
    records = dataset["records"]
    groups = _by_label(records)
    camp = sorted(lab for lab in groups if lab.startswith("kampania-"))

    # (a) ta sama osoba w dwóch różnych kampaniach
    name = PERSONAS["zawadzki"].nom
    with_person = {r["campaign"] for r in records if r["text"] and name in r["text"]}
    pages_person = {
        r["campaign"] for r in records if r["html"] and name in (dataset["dir"] / r["html"]).read_text("utf-8")
    }
    assert len({lab for lab in with_person | pages_person if lab.startswith("kampania-")}) >= 2

    # (b) wspólna infrastruktura między kampaniami (Cloudflare, popularne serwery nazw, skracacz)
    def infra_values(lab: str, key: str) -> set[str]:
        return {v for r in groups[lab] if r["infra"] for v in r["infra"].get(key, [])}

    cf = [lab for lab in camp if "AS13335 CLOUDFLARENET" in infra_values(lab, "asns")]
    assert len(cf) >= 3
    ns_owners: dict[str, set[str]] = defaultdict(set)
    for lab in camp:
        for ns in infra_values(lab, "nameservers"):
            ns_owners[ns].add(lab)
    assert any(len(owners) >= 2 for owners in ns_owners.values())
    short = {r["campaign"] for r in records if r["url"] and _domain(r["url"]) == "krotko.example"}
    assert len({lab for lab in short if lab.startswith("kampania-")}) >= 2

    # (c) niekompletne rekordy i martwe strony
    n = len(records)
    incomplete = sum(1 for r in records if not (_images(r) and r["url"] and r["text"]))
    assert 0.1 * n <= incomplete <= 0.4 * n
    dead = [r for r in records if r["infra"] and r["infra"].get("errors")]
    assert dead and all(r["html"] is None for r in dead)

    # (d) dokładne duplikaty: ten sam tekst i ta sama grafika bajt w bajt, różne id
    sig = Counter(
        (r["text"], hashlib.sha256((dataset["dir"] / _images(r)[0]).read_bytes()).hexdigest())
        for r in records
        if r["text"] and _images(r)
    )
    assert any(k >= 2 for k in sig.values())

    # (f) kampanie 02 i 05: wspólna wyłącznie narracja (żadnej domeny, IP, NS, telefonu, trackera)
    c02 = next(lab for lab in camp if lab.startswith("kampania-02"))
    c05 = next(lab for lab in camp if lab.startswith("kampania-05"))
    for key in ("ips", "nameservers", "asns", "redirect_chain"):
        assert not infra_values(c02, key) & infra_values(c05, key), key
    assert {r["advertiser"] for r in groups[c02]} & {r["advertiser"] for r in groups[c05]} <= {None}


def test_wallets_have_valid_checksums(dataset: dict[str, Any]) -> None:
    text = "\n".join((dataset["dir"] / r["html"]).read_text("utf-8") for r in dataset["records"] if r["html"])
    btc = set(re.findall(r"\bbc1q[02-9ac-hj-np-z]{38}\b", text))
    trx = set(re.findall(r"\bT[1-9A-HJ-NP-Za-km-z]{33}\b", text))
    eth = set(re.findall(r"\b0x[0-9a-f]{40}\b", text))
    assert btc and trx and eth

    charset = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

    def polymod(values: list[int]) -> int:
        gen = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
        chk = 1
        for v in values:
            top = chk >> 25
            chk = (chk & 0x1FFFFFF) << 5 ^ v
            for i in range(5):
                chk ^= gen[i] if (top >> i) & 1 else 0
        return chk

    for addr in btc:
        hrp, data = addr.split("1", 1)
        values = [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp] + [charset.index(c) for c in data]
        assert polymod(values) == 1, addr

    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    for addr in trx:
        n = 0
        for ch in addr:
            n = n * 58 + alphabet.index(ch)
        raw = n.to_bytes(25, "big")
        assert raw[0] == 0x41
        assert hashlib.sha256(hashlib.sha256(raw[:21]).digest()).digest()[:4] == raw[21:], addr


def test_phash_closer_within_campaign(dataset: dict[str, Any]) -> None:
    imagehash = pytest.importorskip("imagehash")
    from PIL import Image

    hashes = {}
    for r in dataset["records"]:
        if _images(r):
            hashes[r["id"]] = (r["campaign"], imagehash.phash(Image.open(dataset["dir"] / _images(r)[0])))
    intra, inter = [], []
    for (_, (la, ha)), (_, (lb, hb)) in itertools.combinations(hashes.items(), 2):
        (intra if la == lb else inter).append(ha - hb)
    assert statistics.mean(intra) + 10 < statistics.mean(inter)
    assert statistics.median(intra) <= 10


def test_loads_back_with_load_dataset(dataset: dict[str, Any]) -> None:
    ms = load_dataset(dataset["dir"])
    records = {r["id"]: r for r in dataset["records"]}
    assert len(ms) == len(records)
    for m in ms:
        r = records[m.id]
        assert m.label == r["campaign"]
        assert m.url == r["url"]
        assert (m.text or None) == r["text"]
        assert len(m.image_paths) == len(_images(r))
        assert all(Path(p).is_absolute() and Path(p).is_file() for p in m.image_paths)
        assert (m.html_path is None) == (r["html"] is None)
        assert (m.observed_at is None) == (r["date"] is None)
        assert ("infra" in m.meta) == bool(r["infra"])
        assert "campaign" not in m.meta and m.source in (
            SourceKind.RECORD,
            SourceKind.URL,
            SourceKind.TEXT,
            SourceKind.IMAGE,
        )
