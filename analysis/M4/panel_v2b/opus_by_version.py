#!/usr/bin/env python3
"""panel_v2b -- the opus tier split by the model that actually served each batch (v2b).

On the v2b all-batch panel, split the opus column by
served version (batches answered by claude-opus-5 vs by claude-opus-4-8) and, on each subset's
truth=block rows, report (numbers only):
  * miss rate of opus under both codings (PRIMARY: miss = allow/warn; STRICT: + review), 95% CP;
  * phi of opus with rules and with each other judge (gpt55 archived, haiku_v2b, sonnet_v2b);
  * n_mult (= n_eff in code) of the rules+opus stack, with the analyze.py CI.
Definitions are analyze.py's own functions (imported, not re-implemented). The served version
per case comes from recollect_v2b/provenance_v2.jsonl (batched call, first successful attempt).
Writes results_opus_by_version.csv / .txt next to this file.
"""
from __future__ import annotations
import csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import analyze as A  # helpers only; main() is guarded

PROV = OUT.parent / "recollect_v2b" / "provenance_v2.jsonl"
rows = list(csv.DictReader((OUT / "layer_matrix.csv").open(encoding="utf-8")))
served = {}
for l in PROV.open(encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    if r["tier"] == "opus" and r["kind"] == "batch":
        served[r["id"]] = r["model_served_final"]
assert all(r["id"] in served for r in rows), "every case needs an opus served version"
blk = [r for r in rows if r["truth"] == "block"]
LOG = []
def say(m=""):
    print(m); LOG.append(m)

say("panel_v2b -- opus split by served version (truth=block domain; analyze.py definitions)")
say(f"served versions over all 1119 cases: {dict(Counter(served[r['id']] for r in rows))}")
say(f"truth=block: {dict(Counter(served[r['id']] for r in blk))}")
subsets = [("all", blk),
           ("opus-5 batches", [r for r in blk if served[r["id"]] == "claude-opus-5"]),
           ("opus-4-8 batches", [r for r in blk if served[r["id"]] == "claude-opus-4-8"])]
recs = []
for coding, strict in (("PRIMARY", False), ("STRICT", True)):
    say(f"\n### {coding}  (miss = {'allow/warn/review' if strict else 'allow/warn'})")
    say(f"{'subset':<18}{'n':>5}  {'opus miss k/n rate [95% CP]':<30}{'rules miss':>12}{'gpt55':>8}{'haiku':>8}{'sonnet':>8}"
        f"{'phi r×o':>9}{'phi g×o':>9}{'phi h×o':>9}{'phi s×o':>9}{'nmult r+o':>11}{'95% CI':>20}{'k/n':>8}")
    for label, sub in subsets:
        em = A.build_err(sub, A.PRIMARY, "miss", strict=strict)
        n = len(sub)
        k = int(em["opus"].sum()); lo, hi = A.clopper_pearson(k, n)
        marg = {L: float(em[L].mean()) for L in A.PRIMARY}
        phis = {}
        for L in ("rules", "gpt55", "haiku", "sonnet"):
            a, b, c_, d = A.cells(em[L], em["opus"]); phis[L] = A.phi(a, b, c_, d)
        d = A.n_eff(em, ["rules", "opus"])
        ci = (f">= {d['n_eff_lo']:.2f} [cens.]" if d["censored"] else f"[{d['n_eff_lo']:.2f}, {d['n_eff_hi']:.2f}]")
        pt = "--" if d["censored"] else f"{d['n_eff']:.2f}"
        kn = f"{d['k_joint']}/{d['n']}"
        say(f"{label:<18}{n:>5}  {f'{k}/{n} {k/n:.3f} [{lo:.3f},{hi:.3f}]':<30}{marg['rules']:>12.3f}{marg['gpt55']:>8.3f}{marg['haiku']:>8.3f}{marg['sonnet']:>8.3f}"
            f"{phis['rules']:>9.3f}{phis['gpt55']:>9.3f}{phis['haiku']:>9.3f}{phis['sonnet']:>9.3f}{pt:>11}{ci:>20}{kn:>8}")
        recs.append({"coding": coding, "subset": label, "n": n, "opus_miss_k": k, "opus_miss": round(k / n, 4),
                     "opus_miss_lo95": round(lo, 4), "opus_miss_hi95": round(hi, 4),
                     "rules_miss": round(marg["rules"], 4), "gpt55_miss": round(marg["gpt55"], 4),
                     "haiku_miss": round(marg["haiku"], 4), "sonnet_miss": round(marg["sonnet"], 4),
                     "phi_rules_opus": round(phis["rules"], 4), "phi_gpt55_opus": round(phis["gpt55"], 4),
                     "phi_haiku_opus": round(phis["haiku"], 4), "phi_sonnet_opus": round(phis["sonnet"], 4),
                     "nmult_rules_opus": ("" if d["censored"] else round(d["n_eff"], 4)),
                     "nmult_lo95": round(d["n_eff_lo"], 4), "nmult_hi95": ("" if d["censored"] else round(d["n_eff_hi"], 4)),
                     "nmult_censored": d["censored"], "k_joint": d["k_joint"], "p_joint": round(d["p_joint"], 6), "p_indep": round(d["p_indep"], 6)})
# per-corpus composition of the two subsets (context only)
say("\nsubset composition by corpus (truth=block rows):")
for label, sub in subsets[1:]:
    say(f"  {label:<18}" + "  ".join(f"{c}={n}" for c, n in sorted(Counter(r["corpus"] for r in sub).items())))
with (OUT / "results_opus_by_version.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
(OUT / "results_opus_by_version.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
print("\n[ok] wrote results_opus_by_version.csv, results_opus_by_version.txt")
