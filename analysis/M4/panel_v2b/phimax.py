#!/usr/bin/env python3
"""phi / phi_max normalisation for the M4 pairwise miss correlations.

phi on two binary indicators cannot reach 1 unless the marginals agree. With p1 <= p2 the
attainable maximum is
    phi_max(p1, p2) = sqrt( p1 (1 - p2) / ( p2 (1 - p1) ) )        (Eq. 8 of arXiv:2608.28327)
attained when erring on the rarer-error layer implies erring on the other. Our rule layer
and the judges have very unequal miss rates, so raw phi alone invites the objection "low
phi because of unequal marginals". This script recomputes every pair from layer_matrix.csv
(pooled, truth=block, n=560) under BOTH miss definitions and writes results_phimax.csv with
phi_max and phi/phi_max beside raw phi. Fisher p and Holm are over the 15 EXTENDED pairs
within each definition (the archived `deepseek` column is faithful-condition and
provenance-incomplete; its pairs are labelled).

Checks: Eq. 8 reproduces the source paper's own example (0.35 / 0.68 -> 0.503); PRIMARY-
definition phi values must equal results_pairs_miss_with_deepseek.csv to 1e-9.
Added 2026-09-03 after the full-text review of 2608.28327; v2 adds the STRICT definition.
"""
import csv, math, statistics, sys
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import analyze as A


def phi_max(pa, pb):
    p1, p2 = min(pa, pb), max(pa, pb)
    return math.sqrt(p1 * (1 - p2) / (p2 * (1 - p1)))


assert abs(phi_max(0.35, 0.68) - 0.503) < 0.001, "Eq. 8 sanity check failed"

rows = list(csv.DictReader((OUT / "layer_matrix.csv").open(encoding="utf-8")))
blk = [r for r in rows if r["truth"] == "block"]
layers = list(A.EXTENDED)
ref = {r["pair"]: float(r["phi"]) for r in csv.DictReader((OUT / "results_pairs_miss_with_deepseek.csv").open(encoding="utf-8"))}

out = []
for dname, strict in (("PRIMARY", False), ("STRICT", True)):
    err = A.build_err(blk, layers, "miss", strict)
    recs, pv = [], []
    for X, Y in combinations(layers, 2):
        a, b, c_, d = A.cells(err[X], err[Y]); n = a + b + c_ + d
        pa, pb = (a + b) / n, (a + c_) / n
        ph = A.phi(a, b, c_, d); pm = phi_max(pa, pb)
        _, p = stats.fisher_exact([[a, b], [c_, d]])
        pv.append(p)
        recs.append({"definition": dname, "pair": f"{X}x{Y}", "group": "rules x judge" if "rules" in (X, Y) else "judge x judge",
                     "n": n, "p_A": pa, "p_B": pb, "phi": ph, "phi_max": pm, "phi_norm": ph / pm, "fisher_p": p,
                     "note": "archived deepseek column (faithful condition, provenance-incomplete)" if "deepseek" in (X, Y) else ""})
    for r, hp in zip(recs, A.holm(pv)):
        r["holm_p"] = hp
        if dname == "PRIMARY":
            assert abs(r["phi"] - ref[r["pair"]]) < 1e-9, f"PRIMARY phi mismatch for {r['pair']}"
    out.extend(recs)

with (OUT / "results_phimax.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

print(f"pooled truth=block n={len(blk)}; PRIMARY phi values match results_pairs_miss_with_deepseek.csv (assert passed)\n")
for dname in ("PRIMARY", "STRICT"):
    print(f"=== {dname} ===")
    print(f"{'pair':<16}{'group':<15}{'p_A':>8}{'p_B':>8}{'phi':>9}{'phi_max':>9}{'phi/max':>9}{'holm_p':>10}  note")
    for r in [x for x in out if x["definition"] == dname]:
        print(f"{r['pair']:<16}{r['group']:<15}{r['p_A']:>8.4f}{r['p_B']:>8.4f}{r['phi']:>+9.4f}{r['phi_max']:>9.4f}{r['phi_norm']:>+9.4f}{r['holm_p']:>10.3g}  {r['note']}")
    print("  group summaries (min / median / max):")
    for grp in ("rules x judge", "judge x judge"):
        for label, sel in (("without archived deepseek", [r for r in out if r["definition"] == dname and r["group"] == grp and not r["note"]]),
                           ("with archived deepseek", [r for r in out if r["definition"] == dname and r["group"] == grp])):
            v = sorted(r["phi_norm"] for r in sel); raw = sorted(r["phi"] for r in sel)
            print(f"    {grp:<15} {label:<26} n={len(v):<3} phi/phi_max {v[0]:+.3f} / {statistics.median(v):+.3f} / {v[-1]:+.3f}"
                  f"   raw phi {raw[0]:+.3f} / {statistics.median(raw):+.3f} / {raw[-1]:+.3f}")
    print()
print("written: results_phimax.csv")
