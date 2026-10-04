#!/usr/bin/env python3
"""M4 step 4 -- fold the re-collected DeepSeek columns into the layer matrix.

Inputs (from recollect_judges.py; each record carries model_served, UTC timestamp, prompt sha,
raw response):
  _recollect_deepseek.jsonl           no-tell  (description blanked)  -> layer `deepseek_nt`
  _recollect_deepseek_faithful.jsonl  faithful (description shown)    -> layer `deepseek_ft`

Outputs:
  _verdicts_deepseek_recollect.jsonl, _verdicts_deepseek_recollect_faithful.jsonl
        archive-schema verdict files (id, risk_level, verdict, confidence, reason); last successful
        record per id wins (resume semantics)
  layer_matrix_recollect.csv    layer_matrix.csv rows of external189 + independent630, + deepseek_nt/_ft
  results_recollect.txt         everything printed below
  results_recollect_pairs_primary.csv / _strict.csv    pairwise table incl. phi_max and phi/phi_max
  results_recollect_provenance.json

Three readings:
  A  version drift, condition-matched: archived `deepseek` (faithful; collected through llm_judge.py's
     prompt, version not recorded) vs `deepseek_ft` (faithful; served version recorded). Condition-
     matched, NOT prompt-identical -- read as an upper bound on drift + prompt wording, not drift alone.
  B  elicitation difference, same model, same hour: `deepseek_nt` vs `deepseek_ft`.
  C  main reading: `deepseek_nt` as a sixth, provenance-clean, cross-vendor layer beside the five
     PRIMARY layers -- pairwise phi / phi_max / c / n_mult under PRIMARY and STRICT miss definitions.
The subset is external189 + independent630 (what was re-collected); PRIMARY pairs are recomputed on
that subset for like-for-like numbers and will differ from the pooled-560 headline table.
Deterministic: exact tests only, no RNG.
"""
from __future__ import annotations
import csv, json, math, sys
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import analyze as A  # helper functions only; main() is guarded

VALID = {"allow", "warn", "block", "review"}
PRIMARY = list(A.PRIMARY)
NT, FT, AR = "deepseek_nt", "deepseek_ft", "deepseek"
SUBSET = ["external189", "independent630"]
say = A.say


def phi_max(pa, pb):
    p1, p2 = min(pa, pb), max(pa, pb)
    if p1 <= 0 or p2 >= 1:
        return float("nan")
    return math.sqrt(p1 * (1 - p2) / (p2 * (1 - p1)))


def parse(fname):
    path = OUT / fname
    if not path.exists():
        return None, {"file": fname, "present": False}
    recs, st, served, status, times, cond = {}, Counter(), Counter(), Counter(), [], None
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        st["records"] += 1
        try:
            r = json.loads(line)
        except Exception:
            st["unreadable_line"] += 1
            continue
        status[str(r.get("http_status"))] += 1
        cond = r.get("condition", cond)
        if r.get("http_status") != 200 or not r.get("parsed_text"):
            st["http_or_empty_fail"] += 1
            continue
        try:
            o = json.loads(r["parsed_text"])
        except Exception:
            st["json_parse_fail"] += 1
            continue
        v = str(o.get("verdict", "")).strip().lower()
        if v not in VALID:
            st["invalid_verdict"] += 1
            continue
        if o.get("id") and str(o["id"]).strip() != r["id"]:
            st["id_echo_mismatch"] += 1
        if r["id"] in recs:
            st["duplicate_id_last_wins"] += 1
        recs[r["id"]] = {"id": r["id"], "corpus": r.get("corpus"),
                         "risk_level": str(o.get("risk_level", "")).strip().lower(), "verdict": v,
                         "confidence": o.get("confidence"), "reason": o.get("reason", "")}
        st["ok"] += 1
        served[r.get("model_served", "")] += 1
        times.append(r.get("collected_at_utc", ""))
    prov = {"file": fname, "present": True, "condition": cond, "records": st["records"],
            "ok_unique_ids": len(recs), "counts": dict(st), "http_status": dict(status),
            "model_served": dict(served), "first_utc": min(times) if times else None,
            "last_utc": max(times) if times else None}
    return recs, prov


def write_verdicts(recs, fname):
    with (OUT / fname).open("w", encoding="utf-8") as fh:
        for r in recs.values():
            fh.write(json.dumps({k: r[k] for k in ("id", "risk_level", "verdict", "confidence", "reason")},
                                ensure_ascii=False) + "\n")


def mcnemar(b, c):
    return 1.0 if b + c == 0 else float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)


def compare(rows, X, Y, label):
    """Two verdict columns on the same cases: agreement, miss rates (block), false alarms (allow)."""
    both = [r for r in rows if r.get(X) and r.get(Y)]
    say(f"\n--- {label}: `{X}` vs `{Y}`  (n = {len(both)} cases with both) ---")
    if not both:
        say("  (nothing to compare)")
        return
    agree = sum(1 for r in both if r[X] == r[Y])
    say(f"  exact 4-way verdict agreement: {agree}/{len(both)} = {agree / len(both):.3f}")
    conf = Counter((r[X], r[Y]) for r in both)
    say("  confusion (rows = " + X + ", cols = " + Y + "):")
    vs = ["allow", "warn", "block", "review"]
    say("            " + "".join(f"{v:>8}" for v in vs))
    for vx in vs:
        say(f"  {vx:>8}  " + "".join(f"{conf[(vx, vy)]:>8}" for vy in vs))
    for corpus in ["POOLED"] + SUBSET:
        sub = both if corpus == "POOLED" else [r for r in both if r["corpus"] == corpus]
        blk = [r for r in sub if r["truth"] == "block"]
        alw = [r for r in sub if r["truth"] == "allow"]
        parts = []
        for dname, strict in (("PRIMARY", False), ("STRICT", True)):
            if not blk:
                continue
            ex = np.array([A.miss(r[X], strict) for r in blk]); ey = np.array([A.miss(r[Y], strict) for r in blk])
            b = int(((ex == 1) & (ey == 0)).sum()); c = int(((ex == 0) & (ey == 1)).sum())
            parts.append(f"{dname} miss {ex.mean():.3f} vs {ey.mean():.3f} (n={len(blk)}, discordant {b}/{c}, McNemar p={mcnemar(b, c):.3f})")
        if alw:
            fx = np.array([A.false_alarm(r[X]) for r in alw]); fy = np.array([A.false_alarm(r[Y]) for r in alw])
            b = int(((fx == 1) & (fy == 0)).sum()); c = int(((fx == 0) & (fy == 1)).sum())
            parts.append(f"false-alarm {fx.mean():.3f} vs {fy.mean():.3f} (n={len(alw)}, discordant {b}/{c}, McNemar p={mcnemar(b, c):.3f})")
        say(f"  [{corpus:<14}] " + " | ".join(parts))


def with_phimax(recs):
    for r in recs:
        pm = phi_max(r["p_A"], r["p_B"])
        r["phi_max"] = pm
        r["phi_norm"] = r["phi"] / pm if pm and not math.isnan(pm) and pm > 0 else float("nan")
    return recs


def group_summary(recs, tag):
    for lbl, sel in (("rules x judge (incl. NT)", [r for r in recs if "rules" in (r["A"], r["B"])]),
                     ("judge x judge (incl. NT)", [r for r in recs if "rules" not in (r["A"], r["B"])]),
                     ("NT pairs only", [r for r in recs if NT in (r["A"], r["B"])])):
        ph = [r["phi"] for r in sel if not math.isnan(r["phi"])]
        pn = [r["phi_norm"] for r in sel if not math.isnan(r["phi_norm"])]
        if ph:
            say(f"  {tag} {lbl:<26} n={len(sel):<3} phi min/med/max {min(ph):+.3f}/{float(np.median(ph)):+.3f}/{max(ph):+.3f}"
                f"   phi/phi_max {min(pn):+.3f}/{float(np.median(pn)):+.3f}/{max(pn):+.3f}"
                f"   sig(Holm .05) {sum(1 for r in sel if r['holm_p'] < .05)}/{len(sel)}")


def nmult_line(err, layers, label):
    d = A.n_eff(err, layers)
    lo, hi = d["n_eff_lo"], d["n_eff_hi"]
    ci = "censored (k=0)" if d["censored"] else f"[{lo:.2f}, {hi:.2f}]"
    say(f"  {label:<34} N={d['N']}  k={d['k_joint']:<3} p_joint={d['p_joint']:.4f} p_indep={d['p_indep']:.4f}"
        f"  n_mult={d['n_eff']:.2f} {ci}")


def main() -> int:
    A.LOG.clear()
    say("M4 step 4 -- re-collected DeepSeek columns (merge_recollect.py)")
    nt, p_nt = parse("_recollect_deepseek.jsonl")
    ft, p_ft = parse("_recollect_deepseek_faithful.jsonl")
    say("\n### provenance")
    for p in (p_nt, p_ft):
        say("  " + json.dumps(p, ensure_ascii=False))
    if nt:
        write_verdicts(nt, "_verdicts_deepseek_recollect.jsonl")
    if ft:
        write_verdicts(ft, "_verdicts_deepseek_recollect_faithful.jsonl")

    rows = [r for r in csv.DictReader((OUT / "layer_matrix.csv").open(encoding="utf-8")) if r["corpus"] in SUBSET]
    for r in rows:
        r[NT] = nt[r["id"]]["verdict"] if nt and r["id"] in nt else ""
        r[FT] = ft[r["id"]]["verdict"] if ft and r["id"] in ft else ""
    say("\n### coverage of the re-collected columns on the layer matrix")
    for corpus in SUBSET:
        sub = [r for r in rows if r["corpus"] == corpus]
        say(f"  {corpus:<15} rows {len(sub):<4} {NT} {sum(1 for r in sub if r[NT]):<4} {FT} {sum(1 for r in sub if r[FT])}")
    fields = list(rows[0].keys())
    with (OUT / "layer_matrix_recollect.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)

    say("\n### verdict distributions on truth=block rows (the review habit matters: see R.4.4)")
    blk_all = [r for r in rows if r["truth"] == "block"]
    for L in PRIMARY + [AR, NT, FT]:
        c = Counter(r[L] for r in blk_all if r.get(L))
        tot = sum(c.values())
        if tot:
            say(f"  {L:<12} n={tot:<4} " + "  ".join(f"{v}={c[v]}" for v in ("allow", "warn", "block", "review")))

    say("\n### A. version drift under a matched condition (archived faithful vs re-collected faithful)")
    say("    caveat: condition-matched, not prompt-identical (archived column came through llm_judge.py's prompt)")
    compare(rows, AR, FT, "A")
    say("\n### B. elicitation difference, same served model (no-tell vs faithful)")
    compare(rows, NT, FT, "B")
    say("\n### A'. archived faithful vs re-collected NO-TELL (what the sensitivity block compared against before)")
    compare(rows, AR, NT, "A'")

    say("\n### C. deepseek_nt as a sixth layer beside the five PRIMARY layers (subset = external189 + independent630)")
    blk = [r for r in rows if r["truth"] == "block" and r[NT]]
    alw = [r for r in rows if r["truth"] == "allow" and r[NT]]
    layers = PRIMARY + [NT]
    if not blk:
        say("  (no block rows with deepseek_nt yet)")
    for dname, strict in (("PRIMARY", False), ("STRICT", True)):
        if not blk:
            break
        err = A.build_err(blk, layers, "miss", strict)
        say(f"\n  miss rates ({dname}): " + "  ".join(f"{L}={err[L].mean():.3f}" for L in layers))
        recs = with_phimax(A.pair_table(err, layers, f"MISS correlation, {dname} definition, truth=block subset"))
        group_summary(recs, dname)
        with (OUT / f"results_recollect_pairs_{dname.lower()}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
        say(f"\n  n_mult ({dname}):")
        nmult_line(err, ["rules", NT], "rules + deepseek_nt")
        for J in ("gpt55", "haiku", "opus", "sonnet"):
            nmult_line(err, [J, NT], f"{J} + deepseek_nt")
        nmult_line(err, ["haiku", "opus", "sonnet"], "3 same-vendor judges (ref)")
        nmult_line(err, ["haiku", "opus", "sonnet", NT], "3 same-vendor + deepseek_nt")
        nmult_line(err, ["gpt55", "haiku", "opus", "sonnet"], "4 judges (ref)")
        nmult_line(err, ["gpt55", "haiku", "opus", "sonnet", NT], "4 judges + deepseek_nt")
        nmult_line(err, ["rules", "gpt55", "haiku", "opus", "sonnet"], "rules + 4 judges (ref)")
        nmult_line(err, ["rules", "gpt55", "haiku", "opus", "sonnet", NT], "rules + 4 judges + deepseek_nt")
        say(f"\n  per-corpus phi for the deepseek_nt pairs ({dname}):")
        for corpus in SUBSET:
            sub = [r for r in blk if r["corpus"] == corpus]
            e = A.build_err(sub, layers, "miss", strict)
            parts = []
            for L in PRIMARY:
                a, b, c_, d = A.cells(e[L], e[NT])
                parts.append(f"{L}x{NT}={A.phi(a, b, c_, d):+.3f}")
            say(f"    {corpus:<15} n={len(sub):<4} " + "  ".join(parts))
    if alw:
        ef = A.build_err(alw, layers, "falsealarm")
        A.pair_table(ef, layers, "FALSE-ALARM correlation, truth=allow subset")

    (OUT / "results_recollect.txt").write_text("\n".join(A.LOG) + "\n", encoding="utf-8")
    (OUT / "results_recollect_provenance.json").write_text(json.dumps({"no_tell": p_nt, "faithful": p_ft}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[ok] wrote results_recollect.txt, results_recollect_pairs_*.csv, results_recollect_provenance.json, layer_matrix_recollect.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
