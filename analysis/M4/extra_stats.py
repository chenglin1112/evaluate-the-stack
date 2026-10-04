#!/usr/bin/env python3
"""Additional statistics on the stacks: floors, per-corpus intervals, bootstrap, homogeneity,
and a second implementation of the difficulty share.

(1) n_mult has a floor above 1 when marginals differ: P_joint <= min_L p_L, so
       n_mult >= ln(min p) / ln g. Reported per stack, with a normalised reading
       (n_mult - floor) / (|S| - floor) in [0, 1] (1 = multiplication rule).
(2) per-corpus n_mult with the Clopper-Pearson interval, both definitions, all ten
       pairs; censored stacks get their lower bound instead of "cens.".
(3) case-level bootstrap intervals for the pooled Table V stacks (B = 4000,
       seed 20260903), with the fraction of censored resamples.
(4) Breslow-Day homogeneity of the per-corpus odds ratios behind the CMH common OR.
(5) difficulty decomposition, SECOND IMPLEMENTATION written without reading
       analyze.decompose: for the six judge pairs, theta_i = mean error of the OTHER
       PRIMARY layers on case i; P_EL = sum over strata of (mean x_A)(mean x_B) weighted
       by stratum size; share = (P_EL - P_indep) / (P_obs - P_indep). Reported with
       (i) probe = other judges only (k = 2), (ii) probe = rules + other judges (k = 3),
       pooled over the six pairs, with a delete-1 jackknife interval. No archived
       DeepSeek column is used anywhere in this file.
Deterministic except the bootstrap, whose seed is fixed. Writes results_extra_stats.txt
and results_extra_stats_*.csv next to this file.
"""
from __future__ import annotations
import csv, math, sys
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import analyze as A   # helpers only (miss, cells, phi, clopper_pearson, n_eff); main() is guarded

PRIMARY = list(A.PRIMARY)
JUDGES = [L for L in PRIMARY if L != "rules"]
CORPORA = ["internal300", "independent630", "external189"]
LOG = []
def say(m=""):
    print(m); LOG.append(m)

rows = list(csv.DictReader((OUT / "layer_matrix.csv").open(encoding="utf-8")))
blk_all = [r for r in rows if r["truth"] == "block"]

def err_of(sub, layers, strict):
    return {L: np.array([A.miss(r[L], strict) for r in sub], dtype=np.int8) for L in layers}

def floor_of(ps):
    g = math.exp(sum(math.log(p) for p in ps) / len(ps))
    return math.log(min(ps)) / math.log(g) if 0 < g < 1 and min(ps) > 0 else float("nan")

def nmult_stack(err, layers):
    d = A.n_eff(err, layers)
    ps = [float(err[L].mean()) for L in layers]
    fl = floor_of(ps)
    N = len(layers)
    norm = (d["n_eff"] - fl) / (N - fl) if not (math.isnan(d["n_eff"]) or math.isnan(fl) or N == fl) else float("nan")
    return d, fl, norm

# ---------------------------------------------------------------- 1 + 2: floors and per-corpus intervals
say("### 1. n_mult floor and normalised reading, pooled (n=560)")
say(f"{'stack':<30}{'def':<8}{'k':>4}{'n_mult':>8}{'lo':>7}{'hi':>7}{'floor':>7}{'norm':>7}")
stacks = [["rules", J] for J in JUDGES] + [list(p) for p in combinations(JUDGES, 2)] + \
         [["haiku", "opus", "sonnet"], JUDGES, PRIMARY]
rec1 = []
for dname, strict in (("PRIMARY", False), ("STRICT", True)):
    err = err_of(blk_all, PRIMARY, strict)
    for S in stacks:
        d, fl, norm = nmult_stack(err, S)
        lo, hi = d["n_eff_lo"], d["n_eff_hi"]
        say(f"{'+'.join(S):<30}{dname:<8}{d['k_joint']:>4}{d['n_eff']:>8.2f}{lo:>7.2f}{hi:>7.2f}{fl:>7.2f}{norm:>7.2f}")
        rec1.append({"stack": "+".join(S), "definition": dname, "k": d["k_joint"], "n_mult": d["n_eff"], "ci_lo": lo, "ci_hi": hi, "floor": fl, "normalised": norm})
with (OUT / "results_extra_stats_floors.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rec1[0])); w.writeheader(); w.writerows(rec1)

say("\n### 2. per-corpus two-layer n_mult with 95% CI, floor, and censored lower bounds")
say(f"{'pair':<16}{'corpus':<16}{'def':<8}{'n':>5}{'k':>4}{'n_mult':>8}{'lo':>7}{'hi':>7}{'floor':>7}")
rec2 = []
for dname, strict in (("PRIMARY", False), ("STRICT", True)):
    for corpus in CORPORA:
        sub = [r for r in blk_all if r["corpus"] == corpus]
        err = err_of(sub, PRIMARY, strict)
        for X, Y in combinations(PRIMARY, 2):
            d, fl, norm = nmult_stack(err, [X, Y])
            val = "cens." if d["censored"] else f"{d['n_eff']:.2f}"
            hi_s = "inf" if math.isinf(d["n_eff_hi"]) else f"{d['n_eff_hi']:.2f}"
            say(f"{X+'x'+Y:<16}{corpus:<16}{dname:<8}{len(sub):>5}{d['k_joint']:>4}{val:>8}{d['n_eff_lo']:>7.2f}{hi_s:>7}{fl:>7.2f}")
            rec2.append({"pair": f"{X}x{Y}", "corpus": corpus, "definition": dname, "n": len(sub), "k": d["k_joint"],
                         "n_mult": d["n_eff"], "ci_lo": d["n_eff_lo"], "ci_hi": d["n_eff_hi"], "floor": fl, "censored": d["censored"]})
with (OUT / "results_extra_stats_percorpus.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rec2[0])); w.writeheader(); w.writerows(rec2)
say("  interval overlap check (STRICT): does any judge-pair CI overlap any rule-pair CI, per corpus?")
for corpus in CORPORA:
    rp = [r for r in rec2 if r["corpus"] == corpus and r["definition"] == "STRICT" and "rules" in r["pair"]]
    jp = [r for r in rec2 if r["corpus"] == corpus and r["definition"] == "STRICT" and "rules" not in r["pair"]]
    ov = sum(1 for a in rp for b in jp if not (a["ci_lo"] > b["ci_hi"] or b["ci_lo"] > a["ci_hi"]))
    say(f"    {corpus:<16} overlapping (rule pair, judge pair) combinations: {ov} of {len(rp)*len(jp)}"
        f"   rule lower bounds {min(r['ci_lo'] for r in rp):.2f}-{max(r['ci_lo'] for r in rp):.2f}; judge point range "
        f"{min((r['n_mult'] for r in jp if not r['censored']), default=float('nan')):.2f}-{max((r['n_mult'] for r in jp if not r['censored']), default=float('nan')):.2f}")

# ---------------------------------------------------------------- 3: bootstrap
say("\n### 3. case-level bootstrap 95% intervals, pooled Table V stacks (B=4000, seed 20260903)")
rng = np.random.default_rng(20260903)
n = len(blk_all); B = 4000
idx = rng.integers(0, n, size=(B, n))
say(f"{'stack':<30}{'def':<8}{'n_mult':>8}{'boot lo':>9}{'boot hi':>9}{'CP lo':>7}{'CP hi':>7}{'cens%':>7}")
rec3 = []
for dname, strict in (("PRIMARY", False), ("STRICT", True)):
    err = err_of(blk_all, PRIMARY, strict)
    M = np.stack([err[L] for L in PRIMARY], axis=1)  # n x 5
    for S in stacks:
        cols = [PRIMARY.index(L) for L in S]
        sub = M[:, cols]
        vals = []
        cens = 0
        for b in range(B):
            X = sub[idx[b]]
            ps = X.mean(axis=0)
            k = int(X.all(axis=1).sum())
            if k == 0 or (ps <= 0).any() or (ps >= 1).any():
                cens += 1; continue
            g = math.exp(float(np.log(ps).mean()))
            vals.append(math.log(k / n) / math.log(g))
        d = A.n_eff(err, S)
        lo, hi = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))
        say(f"{'+'.join(S):<30}{dname:<8}{d['n_eff']:>8.2f}{lo:>9.2f}{hi:>9.2f}{d['n_eff_lo']:>7.2f}{d['n_eff_hi']:>7.2f}{100*cens/B:>6.1f}%")
        rec3.append({"stack": "+".join(S), "definition": dname, "n_mult": d["n_eff"], "boot_lo": lo, "boot_hi": hi,
                     "cp_lo": d["n_eff_lo"], "cp_hi": d["n_eff_hi"], "censored_resamples_pct": 100 * cens / B})
with (OUT / "results_extra_stats_bootstrap.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rec3[0])); w.writeheader(); w.writerows(rec3)

# ---------------------------------------------------------------- 4: Breslow-Day
say("\n### 4. Breslow-Day homogeneity of per-corpus odds ratios (PRIMARY), the assumption behind a common odds ratio")
try:
    from statsmodels.stats.contingency_tables import StratifiedTable
    for X, Y in combinations(PRIMARY, 2):
        tables = []
        for corpus in CORPORA:
            sub = [r for r in blk_all if r["corpus"] == corpus]
            e = err_of(sub, [X, Y], False)
            a, b, c_, d = A.cells(e[X], e[Y])
            tables.append(np.array([[a, b], [c_, d]], dtype=float))
        st = StratifiedTable(tables)
        bd = st.test_equal_odds(adjust=False)
        say(f"  {X+'x'+Y:<16} Breslow-Day stat {bd.statistic:6.2f}  p = {bd.pvalue:.3f}   (per-corpus OR: "
            + ", ".join(f"{(t[0,0]*t[1,1])/(t[0,1]*t[1,0]) if t[0,1]*t[1,0] > 0 else float('inf'):.2f}" for t in tables) + ")")
except Exception as ex:
    say(f"  statsmodels unavailable or failed: {ex!r}; report 'stratified odds ratio' without a homogeneity claim")

# ---------------------------------------------------------------- 5: difficulty decomposition, second implementation
say("\n### 5. difficulty share, second implementation (judge pairs; probe never includes the archived DeepSeek column)")
def share_for(err, pairs, probe_fn):
    num = den = 0.0
    per = []
    for X, Y in pairs:
        xa, xb = err[X].astype(float), err[Y].astype(float)
        th = probe_fn(X, Y)
        p_obs = float((xa * xb).mean()); p_ind = float(xa.mean() * xb.mean())
        p_el = 0.0
        for t in np.unique(th):
            m = th == t
            p_el += m.mean() * xa[m].mean() * xb[m].mean()
        num += p_el - p_ind; den += p_obs - p_ind
        per.append((f"{X}x{Y}", p_ind, p_el, p_obs, (p_el - p_ind) / (p_obs - p_ind) if p_obs != p_ind else float("nan")))
    return num / den if den else float("nan"), per

jpairs = list(combinations(JUDGES, 2))
for dname, strict in (("PRIMARY", False), ("STRICT", True)):
    err = err_of(blk_all, PRIMARY, strict)
    for label, probe_layers_fn in (("probe = other two judges (k=2)", lambda X, Y: [L for L in JUDGES if L not in (X, Y)]),
                                   ("probe = rules + other two judges (k=3)", lambda X, Y: [L for L in PRIMARY if L not in (X, Y)])):
        def probe(X, Y, f=probe_layers_fn):
            Ls = f(X, Y)
            return np.mean([err[L] for L in Ls], axis=0)
        s, per = share_for(err, jpairs, probe)
        # delete-1 jackknife on the pooled share
        jk = []
        for i in range(len(blk_all)):
            keep = np.ones(len(blk_all), dtype=bool); keep[i] = False
            e2 = {L: err[L][keep] for L in PRIMARY}
            def probe2(X, Y, f=probe_layers_fn, e2=e2):
                return np.mean([e2[L] for L in f(X, Y)], axis=0)
            jk.append(share_for(e2, jpairs, probe2)[0])
        jk = np.array(jk); m = len(jk)
        se = math.sqrt((m - 1) / m * float(((jk - jk.mean()) ** 2).sum()))
        say(f"  {dname:<8} {label:<42} pooled share = {100*s:5.1f}%   jackknife SE {100*se:4.1f}pp   approx 95% [{100*(s-1.96*se):5.1f}%, {100*(s+1.96*se):5.1f}%]")
        say("     per pair: " + "  ".join(f"{p[0]} {100*p[4]:4.0f}%" for p in per))

(OUT / "results_extra_stats.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
print("\n[ok] wrote results_extra_stats.txt + results_extra_stats_{floors,percorpus,bootstrap}.csv")
