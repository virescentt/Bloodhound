"""Ewaluacja wyników potoku na zbiorze z etykietami (syntetycznym albo NASK).

    report = evaluate(ws, cfg)      # słownik (JSON) z metrykami, tabelami i listą błędów
    print(format_report(report))    # czytelny raport po polsku

Co mierzymy (kryteria NASK: przypisanie nowych materiałów do znanych grup, nie łączenie niepowiązanych,
odporność na duplikaty i niekompletne dane):

- grupowanie: precyzja/pełność/F1 par, B-Cubed, ARI, NMI (kampanie z ``ws.campaigns()``),
- trafność powiązań: odsetek powiązań łączących materiały o tej samej etykiecie — ogółem, wg relacji
  (duplikat / wariant / powiązany) i wg rodzaju przesłanki (do kalibracji wag),
- przypisanie (najbliższy sąsiad): materiał z etykietą występującą co najmniej 2 razy jest „przypisany
  poprawnie”, gdy jego najsilniejsze powiązanie prowadzi do materiału z tą samą etykietą i ma wynik
  ≥ ``link.assign.min_score`` — przybliżenie leave-one-out analizy nowego materiału,
- przypisanie do kampanii (leave-one-out): kampania z najsilniejszym powiązaniem (bez samego materiału)
  musi mieć tę samą etykietę większościową,
- „nie łączyć”: materiał z unikalną etykietą nie może mieć powiązania o wyniku ≥ progu.

Etykiety typu „-1”, „noise”, „brak” traktujemy jak szum: każdy taki materiał ma własną, unikalną etykietę.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from slad.eval.metrics import ari_nmi, bcubed, pairwise_prf
from slad.models import Link, Material

NOISE_LABELS = frozenset(
    {"-1", "noise", "szum", "none", "null", "nan", "brak", "n/a", "unknown", "nieznana", "inne", "other", "singleton"}
)
DEFAULT_MIN_SCORE = 4.0
TOP_ERRORS = 10


def gold_labels(materials: list[Material]) -> dict[str, str]:
    """id → etykieta wzorcowa (tylko materiały z etykietą); szum dostaje etykietę unikalną."""
    out: dict[str, str] = {}
    for m in materials:
        if m.label is None or not str(m.label).strip():
            continue
        label = str(m.label).strip()
        out[m.id] = f"__szum__:{m.id}" if label.lower() in NOISE_LABELS else label
    return out


def _min_score(cfg: Any) -> float:
    try:
        return float(cfg.link.assign.min_score)
    except AttributeError:
        return DEFAULT_MIN_SCORE


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _relation(lk: Link) -> str:
    rel = lk.relation
    return str(getattr(rel, "value", rel))


def _link_stats(links: list[Link], gold: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluated = correct = 0
    by_rel: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    false_links: list[dict[str, Any]] = []
    for lk in links:
        ga, gb = gold.get(lk.a), gold.get(lk.b)
        if ga is None or gb is None:
            continue
        ok = ga == gb
        evaluated += 1
        correct += ok
        rel = _relation(lk)
        by_rel[rel][0] += 1
        by_rel[rel][1] += ok
        for kind in sorted({e.kind for e in lk.evidence}):
            by_kind[kind][0] += 1
            by_kind[kind][1] += ok
        if not ok:
            top = sorted(lk.evidence, key=lambda e: -e.weight)[:4]
            false_links.append(
                {
                    "a": lk.a,
                    "b": lk.b,
                    "label_a": ga,
                    "label_b": gb,
                    "score": round(lk.score, 3),
                    "relation": rel,
                    "evidence": [f"{e.kind} ({e.weight:.2f})" for e in top],
                }
            )
    false_links.sort(key=lambda x: -x["score"])

    def table(d: dict[str, list[int]]) -> dict[str, dict[str, Any]]:
        return {k: {"n": n, "correct": c, "precision": _ratio(c, n)} for k, (n, c) in sorted(d.items())}

    stats = {
        "total": len(links),
        "evaluated": evaluated,
        "correct": correct,
        "precision": _ratio(correct, evaluated),
        "by_relation": table(by_rel),
        "by_evidence": table(by_kind),
    }
    return stats, false_links


def _neighbours(links: list[Link]) -> dict[str, list[tuple[float, str]]]:
    nb: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for lk in links:
        if lk.a == lk.b:
            continue
        nb[lk.a].append((lk.score, lk.b))
        nb[lk.b].append((lk.score, lk.a))
    for v in nb.values():
        v.sort(key=lambda t: (-t[0], t[1]))  # najsilniejsze pierwsze, remis → mniejszy identyfikator
    return nb


def evaluate(ws: Any, cfg: Any = None) -> dict[str, Any]:
    """Pełna ewaluacja katalogu roboczego (``slad.store.Workspace``) — wynik do zapisania jako JSON."""
    materials = ws.materials()
    gold = gold_labels(materials)
    label_count = Counter(gold.values())
    campaigns = ws.campaigns()
    links = ws.links()
    min_score = _min_score(cfg)

    pred: dict[str, str] = {}
    multi = 0
    for c in campaigns:
        for mid in c.members:
            if mid in pred:
                multi += 1  # materiał w kilku kampaniach — liczymy pierwszą
                continue
            pred[mid] = c.id
    members: dict[str, list[str]] = defaultdict(list)
    for mid, cid in pred.items():
        members[cid].append(mid)

    clustering = {"pairwise": pairwise_prf(pred, gold), "bcubed": bcubed(pred, gold), **ari_nmi(pred, gold)}
    link_stats, false_links = _link_stats(links, gold)
    nb = _neighbours(links)

    # przypisanie: najbliższy sąsiad
    eligible = sorted(mid for mid, g in gold.items() if label_count[g] >= 2)
    nn_ok, no_link, below, wrong = [], [], [], []
    for mid in eligible:
        best = nb.get(mid)
        if not best:
            no_link.append(mid)
        elif best[0][0] < min_score:
            below.append(mid)
        elif gold.get(best[0][1]) != gold[mid]:
            wrong.append(
                {
                    "id": mid,
                    "nearest": best[0][1],
                    "score": round(best[0][0], 3),
                    "label": gold[mid],
                    "label_nearest": gold.get(best[0][1]),
                }
            )
        else:
            nn_ok.append(mid)

    # przypisanie do kampanii (leave-one-out): kampania najsilniejszego powiązania, etykieta większościowa bez mid
    camp_ok, camp_unassigned, camp_wrong = 0, 0, []
    for mid in eligible:
        scores: dict[str, float] = {}
        for score, other in nb.get(mid, []):
            cid = pred.get(other)
            if cid is not None and score > scores.get(cid, float("-inf")):
                scores[cid] = score
        if not scores or max(scores.values()) < min_score:
            camp_unassigned += 1
            continue
        cid = min(scores, key=lambda k: (-scores[k], k))
        majority = Counter(gold[x] for x in members[cid] if x != mid and x in gold).most_common(1)
        if majority and majority[0][0] == gold[mid]:
            camp_ok += 1
        else:
            camp_wrong.append(
                {"id": mid, "campaign": cid, "label": gold[mid], "campaign_label": majority[0][0] if majority else None}
            )

    # nie łączyć: materiały z unikalną etykietą
    unique = sorted(mid for mid, g in gold.items() if label_count[g] == 1)
    violations = []
    for mid in unique:
        best = nb.get(mid)
        if best and best[0][0] >= min_score:
            violations.append(
                {"id": mid, "other": best[0][1], "score": round(best[0][0], 3), "label_other": gold.get(best[0][1])}
            )
    in_campaign = [mid for mid in unique if mid in pred and len(members[pred[mid]]) >= 2]

    # tabele kampanii i etykiet
    camp_rows = []
    for c in campaigns:
        labeled = [gold[m] for m in members.get(c.id, []) if m in gold]
        top = Counter(labeled).most_common(1)
        camp_rows.append(
            {
                "id": c.id,
                "name": c.name,
                "size": len(c.members),
                "labeled": len(labeled),
                "majority_label": top[0][0] if top else None,
                "purity": _ratio(top[0][1], len(labeled)) if top else None,
                "labels": len(set(labeled)),
            }
        )
    label_rows = []
    by_label: dict[str, list[str]] = defaultdict(list)
    for mid, g in gold.items():
        by_label[g].append(mid)
    for label, mids in sorted(by_label.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        cover = Counter(pred[m] for m in mids if m in pred)
        best_c = min(cover, key=lambda k: (-cover[k], k)) if cover else None
        label_rows.append(
            {
                "label": label,
                "size": len(mids),
                "campaigns": len(cover),
                "unclustered": sum(1 for m in mids if m not in pred),
                "best_campaign": best_c,
                "best_overlap": cover[best_c] if best_c else 0,
                "recall": _ratio(cover[best_c], len(mids)) if best_c else 0.0,
            }
        )
    merged = [r for r in camp_rows if r["labels"] >= 2]
    split = [r for r in label_rows if r["size"] >= 2 and (r["campaigns"] >= 2 or r["unclustered"] > 0)]

    return {
        "counts": {
            "materials": len(materials),
            "labeled": len(gold),
            "labels": len(label_count),
            "unique_labels": sum(1 for v in label_count.values() if v == 1),
            "campaigns": len(campaigns),
            "clustered": len(pred),
            "links": len(links),
            "multi_membership": multi,
        },
        "clustering": clustering,
        "links": link_stats,
        "assignment": {
            "min_score": min_score,
            "eligible": len(eligible),
            "correct": len(nn_ok),
            "accuracy": _ratio(len(nn_ok), len(eligible)),
            "no_link": len(no_link),
            "below_threshold": len(below),
            "wrong": wrong[:TOP_ERRORS],
            "wrong_count": len(wrong),
        },
        "assignment_campaign": {
            "eligible": len(eligible),
            "correct": camp_ok,
            "accuracy": _ratio(camp_ok, len(eligible)),
            "unassigned": camp_unassigned,
            "wrong": camp_wrong[:TOP_ERRORS],
        },
        "should_not_link": {
            "min_score": min_score,
            "eligible": len(unique),
            "correct": len(unique) - len(violations),
            "accuracy": _ratio(len(unique) - len(violations), len(unique)),
            "violations": violations[:TOP_ERRORS],
            "violation_count": len(violations),
            "in_campaign": len(in_campaign),
        },
        "campaigns": camp_rows,
        "labels": label_rows,
        "errors": {
            "false_links": false_links[:TOP_ERRORS],
            "false_link_count": len(false_links),
            "merged_campaigns": merged[:TOP_ERRORS],
            "split_labels": split[:TOP_ERRORS],
        },
    }


# ---------------------------------------------------------------- raport tekstowy


def _num(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}".replace(".", ",")


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%".replace(".", ",")


def _short(label: str | None) -> str:
    if label is None:
        return "(bez etykiety)"
    return "szum" if label.startswith("__szum__:") else label


def format_report(report: dict[str, Any]) -> str:
    """Raport ewaluacji po polsku (tekst do konsoli / UI)."""
    c = report.get("counts", {})
    cl = report.get("clustering", {})
    lk = report.get("links", {})
    asg = report.get("assignment", {})
    asc = report.get("assignment_campaign", {})
    snl = report.get("should_not_link", {})
    err = report.get("errors", {})
    pw, b3 = cl.get("pairwise", {}), cl.get("bcubed", {})
    out = [
        "EWALUACJA — Ślad Fejka",
        "=" * 60,
        f"Materiały: {c.get('materials', 0)} (z etykietą: {c.get('labeled', 0)}; etykiet: {c.get('labels', 0)}, "
        f"w tym unikalnych: {c.get('unique_labels', 0)})",
        f"Wykryte kampanie: {c.get('campaigns', 0)} (materiałów w kampaniach: {c.get('clustered', 0)}) · "
        f"powiązania: {c.get('links', 0)}",
        "",
        "Grupowanie w kampanie",
        f"  pary        precyzja {_num(pw.get('precision'))} · pełność {_num(pw.get('recall'))} · F1 {_num(pw.get('f1'))}",
        f"  B-Cubed     precyzja {_num(b3.get('precision'))} · pełność {_num(b3.get('recall'))} · F1 {_num(b3.get('f1'))}",
        f"  ARI {_num(cl.get('ari'))} · NMI {_num(cl.get('nmi'))}",
        "",
        "Powiązania (czy łączą materiały tej samej kampanii)",
        f"  trafność ogółem: {_pct(lk.get('precision'))} ({lk.get('correct', 0)}/{lk.get('evaluated', 0)} ocenionych, "
        f"wszystkich {lk.get('total', 0)})",
    ]
    for rel, row in lk.get("by_relation", {}).items():
        out.append(f"    {rel:<12} {_pct(row['precision']):>7}  ({row['correct']}/{row['n']})")
    weak = [(k, r) for k, r in lk.get("by_evidence", {}).items() if r["precision"] is not None and r["precision"] < 0.9]
    if weak:
        out.append("  przesłanki o niskiej trafności (do obniżenia wagi):")
        for kind, row in sorted(weak, key=lambda kr: kr[1]["precision"]):
            out.append(f"    {kind:<18} {_pct(row['precision']):>7}  ({row['correct']}/{row['n']} powiązań)")
    out += [
        "",
        f"Przypisanie nowego materiału (najbliższy sąsiad, próg {_num(asg.get('min_score'), 1)})",
        f"  trafność: {_pct(asg.get('accuracy'))} ({asg.get('correct', 0)}/{asg.get('eligible', 0)}) · bez powiązań: "
        f"{asg.get('no_link', 0)} · poniżej progu: {asg.get('below_threshold', 0)} · do złej kampanii: "
        f"{asg.get('wrong_count', 0)}",
        f"  do kampanii (leave-one-out): {_pct(asc.get('accuracy'))} ({asc.get('correct', 0)}/{asc.get('eligible', 0)}) · "
        f"nieprzypisane: {asc.get('unassigned', 0)}",
        "",
        "Nie łączyć niepowiązanych (materiały z unikalną etykietą)",
        f"  trafność: {_pct(snl.get('accuracy'))} ({snl.get('correct', 0)}/{snl.get('eligible', 0)}) · "
        f"w kampaniach mimo to: {snl.get('in_campaign', 0)}",
    ]
    for v in snl.get("violations", [])[:5]:
        out.append(f"    {v['id']} ↔ {v['other']} (wynik {_num(v['score'], 2)}, {_short(v.get('label_other'))})")
    if err.get("false_links"):
        out += ["", f"Błędne powiązania (najsilniejsze z {err.get('false_link_count', 0)})"]
        for f in err["false_links"][:5]:
            out.append(
                f"  {f['a']} ↔ {f['b']}  wynik {_num(f['score'], 2)} [{f['relation']}]  "
                f"{_short(f['label_a'])} / {_short(f['label_b'])}  ← {', '.join(f['evidence'])}"
            )
    if err.get("merged_campaigns"):
        out += ["", "Kampanie łączące różne etykiety"]
        for r in err["merged_campaigns"][:5]:
            out.append(
                f"  {r['id']} „{r['name']}” — {r['size']} mat., etykiet: {r['labels']}, czystość {_pct(r['purity'])}"
            )
    if err.get("split_labels"):
        out += ["", "Etykiety rozbite na kilka kampanii lub niepogrupowane"]
        for r in err["split_labels"][:5]:
            out.append(
                f"  {_short(r['label'])} — {r['size']} mat. w {r['campaigns']} kampaniach, poza kampaniami: "
                f"{r['unclustered']}, pokrycie najlepszej: {_pct(r['recall'])}"
            )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    """``python -m slad.eval.run [katalog_roboczy] [--json raport.json]``"""
    from slad.settings import load_settings
    from slad.store import Workspace

    ap = argparse.ArgumentParser(description="Ewaluacja grupowania na zbiorze z etykietami")
    ap.add_argument("workspace", nargs="?", help="katalog roboczy (domyślnie z konfiguracji)")
    ap.add_argument("--json", type=Path, help="zapisz pełny raport JSON")
    args = ap.parse_args(argv)
    cfg = load_settings()
    ws = Workspace(Path(args.workspace) if args.workspace else cfg.path(cfg.workspace))
    report = evaluate(ws, cfg)
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    sys.stdout.write(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
