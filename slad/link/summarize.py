"""Opis kampanii: nazwa po polsku, spójność, najczęstsze osoby / marki / narracje / domeny / wskaźniki,
okres aktywności i kluczowe przesłanki (najsilniejsze wspólne ślady w powiązaniach wewnątrz kampanii)."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime

from slad.link.indicators import KIND_GROUP, chain_roles, display_value, indicators, load_stoplists
from slad.models import Campaign, CampaignSummary, Evidence, FeatureSet, Link

log = logging.getLogger(__name__)

TOP_N = 5
TOP_INDICATORS = 10
TOP_KEYWORDS = 10
TOP_EVIDENCE = 8
NAME_SHARE = 0.25  # część materiałów, w której musi wystąpić wartość, by trafić do nazwy
MAX_NAME = 90

# wskaźniki infrastruktury i kontaktu pokazywane w podsumowaniu (domeny osobno)
SUMMARY_INDICATOR_KINDS = (
    "tracker_id",
    "crypto_wallet",
    "phone",
    "email",
    "telegram",
    "whatsapp",
    "iban",
    "short_url",
    "host",
    "form_action",
    "script_domain",
    "favicon",
    "tls_fingerprint",
    "cert_domain",
    "form_fields",
    "title",
    "ip",
    "nameserver",
    "asn",
)

# opisy zbiorcze przesłanek bez wartości (cechy ciągłe)
_AGG_LABELS = {
    "text_minhash": "niemal identyczne teksty",
    "text_tfidf": "podobna treść",
    "image_phash": "te same lub przerobione grafiki",
    "image_clip": "podobne grafiki (CLIP)",
    "face": "ta sama twarz",
    "dom_simhash": "ten sam szablon strony",
    "registrar_created": "domeny rejestrowane u tego samego rejestratora w krótkim odstępie",
    "keyword": "wspólne słowa kluczowe",
}


def _top(counter: Counter, display: Mapping[str, str], n: int) -> list[tuple[str, int]]:
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return [(display.get(v, v), c) for v, c in items]


def _leader(counter: Counter, display: Mapping[str, str], n_members: int, min_count: int = 1) -> str | None:
    if not counter:
        return None
    value, count = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    if count < max(min_count, math.ceil(NAME_SHARE * n_members)):
        return None
    return display.get(value, value)


def _join_pl(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " i " + parts[-1]


def _strip_note(detail: str) -> str:
    return detail.split(" — ")[0]


def summarize(
    cid: str,
    members: Iterable[str],
    features_by_id: Mapping[str, FeatureSet],
    links: Iterable[Link],
    dates: Mapping[str, datetime | None] | None = None,
    indicators_by_id: Mapping[str, Mapping[str, Iterable[str]]] | None = None,
) -> Campaign:
    """Campaign z automatyczną nazwą, spójnością (średni wynik powiązań wewnętrznych) i podsumowaniem.

    `indicators_by_id` (opcjonalnie) — gotowe wskaźniki materiałów (np. z LinkContext), bez ponownej ekstrakcji.
    """
    members = sorted(dict.fromkeys(members))
    mset = set(members)
    internal = [lk for lk in links if lk.a in mset and lk.b in mset and lk.a != lk.b]
    cohesion = round(sum(lk.score for lk in internal) / len(internal), 3) if internal else 0.0

    stop = load_stoplists()
    persons, brands, narratives, domains, keywords, infra, landing = (Counter() for _ in range(7))
    redirectors: set[str] = set()  # domeny pośrednie (skracacze, cloakery) nie nazywają kampanii
    display: dict[str, str] = {}
    for mid in members:
        fs = features_by_id.get(mid)
        if fs is None:
            continue
        try:
            ind = indicators_by_id.get(mid) if indicators_by_id else None
            if ind is None:
                ind = indicators(fs, stop)
        except Exception as e:  # noqa: BLE001
            log.warning("Podsumowanie %s: wskaźniki materiału %s niedostępne (%s)", cid, mid, e)
            continue
        for kind, counter in (
            ("person", persons),
            ("brand", brands),
            ("narrative", narratives),
            ("domain", domains),
            ("keyword", keywords),
        ):
            for v in ind.get(kind, ()):
                counter[v] += 1
                if v not in display:
                    display[v] = display_value(kind, v, fs)
        for kind in SUMMARY_INDICATOR_KINDS:
            for v in ind.get(kind, ()):
                infra[f"{kind}:{v}"] += 1
        try:
            land, via = chain_roles(fs, stop)
        except Exception:  # noqa: BLE001
            land, via = "", set()
        if land:
            landing[land] += 1
        redirectors |= via

    shared_infra = Counter({k: c for k, c in infra.items() if c >= 2})
    seen = [d for d in (dates.get(m) for m in members) if d is not None] if dates else []
    summary = CampaignSummary(
        persons=_top(persons, display, TOP_N),
        brands=_top(brands, display, TOP_N),
        narratives=_top(narratives, display, TOP_N),
        domains=_top(domains, display, TOP_N),
        indicators=_top(shared_infra, {}, TOP_INDICATORS),
        keywords=[
            display.get(v, v) for v, _ in sorted(keywords.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_KEYWORDS]
        ],
        first_seen=min(seen) if seen else None,
        last_seen=max(seen) if seen else None,
    )

    n = len(members)
    who = _leader(persons, display, n) or _leader(brands, display, n)
    story = _leader(narratives, display, n)
    if story and story[:1].isupper() and not story[1:2].isupper():
        story = story[:1].lower() + story[1:]  # „Inwestycja w kryptowaluty” → „inwestycja w kryptowaluty”
    for r in redirectors:
        landing.pop(r, None)
        domains.pop(r, None)
    site = _leader(landing, {}, n, min_count=2 if n >= 2 else 1) or _leader(domains, display, n, min_count=2)
    parts = [p for p in (who, story, site) if p]
    if not parts:
        kws = [display.get(v, v) for v, c in sorted(keywords.items(), key=lambda kv: (-kv[1], kv[0]))[:2] if c >= 2]
        parts = kws
    name = " · ".join(parts) if parts else f"Kampania {cid}"
    if len(name) > MAX_NAME:
        name = name[: MAX_NAME - 1].rstrip() + "…"

    return Campaign(
        id=cid,
        name=name,
        members=members,
        cohesion=cohesion,
        summary=summary,
        key_evidence=key_evidence(internal),
    )


def key_evidence(internal_links: Iterable[Link], top: int = TOP_EVIDENCE) -> list[Evidence]:
    """Najsilniejsze wspólne ślady: (rodzaj, wartość) zsumowane po powiązaniach wewnątrz kampanii."""
    agg: dict[tuple[str, str | None], list] = {}  # klucz → [suma wag, liczba powiązań, najsilniejsza przesłanka]
    for lk in internal_links:
        for e in lk.evidence:
            key = (e.kind, e.value if e.kind not in _AGG_LABELS else None)
            slot = agg.get(key)
            if slot is None:
                agg[key] = [e.weight, 1, e]
            else:
                slot[0] += e.weight
                slot[1] += 1
                if e.weight > slot[2].weight:
                    slot[2] = e
    ranked = sorted(agg.items(), key=lambda kv: (-kv[1][0], kv[0][0], kv[0][1] or ""))[:top]
    out = []
    for (kind, value), (total, count, best) in ranked:
        base = _AGG_LABELS.get(kind) or _strip_note(best.detail)
        times = "powiązaniu" if count == 1 else "powiązaniach"
        out.append(
            Evidence(
                kind=kind,
                group=KIND_GROUP.get(kind, best.group),
                detail=f"{base} — w {count} {times}",
                value=value,
                similarity=None,
                weight=round(total, 3),
            )
        )
    return out
