#!/usr/bin/env python3
"""M4 step 2 -- are the defence layers' errors independent?

Reads layer_matrix.csv (from build_matrix.py) and computes:
  * pairwise error correlation phi / Cohen's kappa / coupling ratio c
  * n_eff, the effective number of independent layers in a stack
  * a two-factor decomposition of the excess over the multiplication rule into
      (D) common difficulty  -- Eckhardt & Lee 1985: difficulty varies across the
          input space, so independently built layers fail together with no shared
          mechanism at all
      (M) shared mechanism   -- a concealment device on the case itself defeats
          several layers by one route

ELICITATION DISCIPLINE (M4 spec item 1). The primary layer set is
    rules, gpt55, haiku, opus, sonnet
because those five are the only ones measured under ONE condition: judge role,
no-tell (the human-written `description` is blanked) for the LLM layers, and the
deterministic PolicyEngine.default() for rules. The `deepseek` column in the
frozen truth CSVs was produced by LLMJudge._build_prompt, which passes
action.description through -- i.e. the FAITHFUL condition. Mixing it with the
no-tell judges would confound coupling with the elicitation difference, so it is
demoted to a clearly-labelled sensitivity block and never enters a headline
number.

Symbol discipline (M4 spec item 2): the common-cause factor is written `c`;
`beta` is reserved for judge miss rates in the cited literature and is not used.

All statistics are deterministic -- exact tests (Fisher, Clopper-Pearson) and
delete-1 jackknife. No RNG, no seeds.
"""
from __future__ import annotations

import csv
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent

PRIMARY = ["rules", "gpt55", "haiku", "opus", "sonnet"]
EXTENDED = ["rules", "deepseek", "gpt55", "haiku", "opus", "sonnet"]
CORPORA = ["internal300", "independent630", "external189"]
MECH_FAMILIES = ["encoded", "indirection", "eval_exec", "remote_exec", "unicode_trick"]
# Concealment devices only: the payload is never spelled out literally. This is
# the narrow reading of "shared mechanism". `eval_exec` / `remote_exec` are LOUD
# signatures (a literal `| sh`, a literal `eval`) -- they are delivery routes,
# not concealment, and every layer is trained to see them, so folding them into
# the mechanism variable would test the wrong hypothesis.
CONCEAL = ["encoded", "indirection"]

LOG: list[str] = []


def say(m: str = "") -> None:
    print(m)
    LOG.append(m)


# --------------------------------------------------------------------------
# error definitions
# --------------------------------------------------------------------------
# MISS -- the security-critical direction, and the only one the "layers
#   multiply" claim is about: a stack fails only if EVERY layer misses.
#   Domain truth=block. `review` counts as CAUGHT (it escalates instead of
#   executing), matching this repo's existing FN definition; the
#   review-as-miss variant is reported as a sensitivity check.
# FALSE ALARM -- composes DISJUNCTIVELY (any layer firing raises the alarm), so
#   stacking makes this direction monotonically worse. Domain truth=allow.
# truth=warn belongs to neither domain and is excluded from both.
def miss(v: str, strict: bool = False) -> int:
    return int(v in (("allow", "warn", "review") if strict else ("allow", "warn")))


def false_alarm(v: str) -> int:
    return int(v != "allow")


# --------------------------------------------------------------------------
def cells(x, y):
    return (int(((x == 1) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum()),
            int(((x == 0) & (y == 1)).sum()), int(((x == 0) & (y == 0)).sum()))


def phi(a, b, c_, d):
    den = math.sqrt((a + b) * (c_ + d) * (a + c_) * (b + d))
    return float("nan") if den == 0 else (a * d - b * c_) / den


def kappa(a, b, c_, d):
    n = a + b + c_ + d
    if n == 0:
        return float("nan")
    po = (a + d) / n
    pe = ((a + b) * (a + c_) + (c_ + d) * (b + d)) / (n * n)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def holm(p):
    m = len(p)
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: p[j])):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def mh_odds_ratio(tables):
    """Cochran-Mantel-Haenszel common OR + two-sided p (chi-square, 1 df)."""
    num = den = s_a = s_ea = s_va = 0.0
    for a, b, c_, d in tables:
        n = a + b + c_ + d
        if n == 0:
            continue
        num += a * d / n
        den += b * c_ / n
        r1, r2, col1 = a + b, c_ + d, a + c_
        s_a += a
        s_ea += r1 * col1 / n
        if n > 1:
            s_va += r1 * r2 * col1 * (n - col1) / (n * n * (n - 1))
    orv = num / den if den > 0 else (float("inf") if num > 0 else float("nan"))
    if s_va <= 0:
        return orv, float("nan")
    return orv, float(stats.chi2.sf((abs(s_a - s_ea) - 0.5) ** 2 / s_va, 1))


def n_eff(err, layers):
    """n_eff = ln P_joint / ln g, g = geometric mean of the per-layer error rates.

    Independent layers give n_eff = N; perfectly coupled layers give 1. The CI
    propagates the exact (Clopper-Pearson) interval on P_joint only, which
    dominates; marginal sampling error is not included."""
    n = len(err[layers[0]])
    ps = [float(err[L].mean()) for L in layers]
    j = np.ones(n, dtype=bool)
    for L in layers:
        j &= err[L] == 1
    k = int(j.sum())
    g = float(np.exp(np.mean(np.log(ps)))) if all(p > 0 for p in ps) else 0.0

    def to(pj):
        return math.log(pj) / math.log(g) if (pj > 0 and 0 < g < 1) else float("nan")

    lo, hi = clopper_pearson(k, n)
    return {"N": len(layers), "n": n, "k_joint": k, "p_joint": k / n,
            "p_indep": float(np.prod(ps)), "g": g,
            "n_eff": to(k / n) if k else float("nan"),
            "n_eff_lo": to(hi), "n_eff_hi": (to(lo) if k else float("inf")),
            "censored": k == 0}


# --------------------------------------------------------------------------
# two-factor decomposition
# --------------------------------------------------------------------------
def theta_leave_both_out(err, layers, A, B):
    """Per-case difficulty from the layers that are NOT A and B, so it carries
    no information about the A-B joint outcome."""
    others = [L for L in layers if L not in (A, B)]
    return np.mean([err[L] for L in others], axis=0)


def decompose(err, layers, A, B):
    """P_indep (multiplication rule) | P_EL (conditional independence GIVEN
    difficulty -- Eckhardt-Lee) | P_obs. P_EL uses each layer's OWN within-
    stratum rate and no joint information."""
    xa, xb = err[A], err[B]
    th = theta_leave_both_out(err, layers, A, B)
    pred = np.zeros(len(th))
    for t in np.unique(th):
        m = th == t
        pred[m] = xa[m].mean() * xb[m].mean()
    p_obs, p_ind, p_el = float((xa & xb).mean()), float(xa.mean() * xb.mean()), float(pred.mean())
    return {"pair": f"{A}x{B}", "p_indep": p_ind, "p_el": p_el, "p_obs": p_obs,
            "excess_total": p_obs - p_ind, "share_difficulty": p_el - p_ind,
            "share_residual": p_obs - p_el}


def jackknife_z(err, layers, A, B):
    """Delete-1 jackknife z for the residual. Deterministic."""
    n = len(err[A])
    full = decompose(err, layers, A, B)["share_residual"]
    idx = np.arange(n)
    v = np.array([decompose({L: err[L][idx != i] for L in layers}, layers, A, B)["share_residual"]
                  for i in range(n)])
    se = math.sqrt((n - 1) / n * float(((v - v.mean()) ** 2).sum()))
    return full, (full / se if se > 0 else float("nan"))


# --------------------------------------------------------------------------
def build_err(rows, layers, kind, strict=False):
    f = (lambda v: miss(v, strict)) if kind == "miss" else false_alarm
    return {L: np.array([f(r[L]) for r in rows], dtype=np.int8) for L in layers}


def pair_table(err, layers, title, csv_name=None):
    pairs = list(combinations(layers, 2))
    n = len(err[layers[0]])
    say(f"\n{title}   (n = {n} cases, {len(pairs)} pairs)")
    say("-" * 106)
    say(f"{'pair':<20}{'p_A':>7}{'p_B':>7}{'P_obs':>8}{'P_ind':>8}{'c':>7}"
        f"{'phi':>8}{'kappa':>8}{'OR':>9}{'Fisher p':>11}{'Holm p':>10}")
    recs, pv = [], []
    for A, B in pairs:
        a, b, c_, d = cells(err[A], err[B])
        m = a + b + c_ + d
        pA, pB = (a + b) / m, (a + c_) / m
        _, p = stats.fisher_exact([[a, b], [c_, d]])
        recs.append({"pair": f"{A}x{B}", "A": A, "B": B, "n": m, "a": a, "b": b, "c": c_, "d": d,
                     "p_A": pA, "p_B": pB, "P_obs": a / m, "P_indep": pA * pB,
                     "c_ratio": (a / m) / (pA * pB) if pA * pB > 0 else float("nan"),
                     "phi": phi(a, b, c_, d), "kappa": kappa(a, b, c_, d),
                     "odds_ratio": (a * d) / (b * c_) if b * c_ else float("inf"),
                     "fisher_p": p})
        pv.append(p)
    for r, ap in zip(recs, holm(pv)):
        r["holm_p"] = ap
        st = "***" if ap < .001 else "**" if ap < .01 else "*" if ap < .05 else ""
        say(f"{r['pair']:<20}{r['p_A']:>7.3f}{r['p_B']:>7.3f}{r['P_obs']:>8.3f}{r['P_indep']:>8.3f}"
            f"{r['c_ratio']:>7.2f}{r['phi']:>8.3f}{r['kappa']:>8.3f}{r['odds_ratio']:>9.2f}"
            f"{r['fisher_p']:>11.2e}{ap:>9.2e}{st}")
    rr = [r for r in recs if r["A"] == "rules" or r["B"] == "rules"]
    ll = [r for r in recs if r not in rr]
    for lbl, grp in (("rules x LLM", rr), ("LLM x LLM ", ll)):
        ph = [r["phi"] for r in grp if not math.isnan(r["phi"])]
        if ph:
            say(f"  {lbl} ({len(grp)} pairs): phi min {min(ph):+.3f} median {float(np.median(ph)):+.3f} "
                f"max {max(ph):+.3f} | {sum(1 for r in grp if r['holm_p'] < .05)} significant (Holm .05)")
    if csv_name:
        with (OUT / csv_name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0]))
            w.writeheader()
            w.writerows(recs)
    return recs


def mechanism_block(err, layers, rows, flag, flag_name):
    pairs = list(combinations(layers, 2))
    M = np.array([r[flag] for r in rows], dtype=np.int8)
    say(f"\n  variable `{flag_name}`: present on {int(M.sum())}/{len(rows)} cases")
    say(f"{'pair':<20}{'MH-OR (M x joint | theta)':>28}{'MH p':>10}{'joint|M=1':>12}{'joint|M=0':>12}")
    recs = []
    for A, B in pairs:
        joint = (err[A] & err[B]).astype(np.int8)
        th = theta_leave_both_out(err, layers, A, B)
        orv, p = mh_odds_ratio([cells(M[th == t], joint[th == t]) for t in np.unique(th)])
        j1 = float(joint[M == 1].mean()) if (M == 1).any() else float("nan")
        j0 = float(joint[M == 0].mean()) if (M == 0).any() else float("nan")
        recs.append({"pair": f"{A}x{B}", "flag": flag_name,
                     "has_rules": int("rules" in (A, B)), "MH_OR": orv, "MH_p": p,
                     "joint_given_M1": j1, "joint_given_M0": j0})
        say(f"{A + 'x' + B:<20}{orv:>28.2f}{p:>10.3f}{j1:>12.3f}{j0:>12.3f}")
    fin = [r["MH_OR"] for r in recs if math.isfinite(r["MH_OR"])]
    llm = [r for r in recs if not r["has_rules"] and math.isfinite(r["MH_OR"])]
    say(f"  -> all pairs: median MH-OR {np.median(fin):.2f}, "
        f"{sum(1 for r in recs if r['MH_p'] < .05)}/{len(recs)} significant at .05")
    if llm:
        say(f"  -> LLM-only pairs (the honest test; see caveat): median MH-OR "
            f"{np.median([r['MH_OR'] for r in llm]):.2f}, "
            f"{sum(1 for r in llm if r['MH_p'] < .05)}/{len(llm)} significant at .05")
    return recs


# ==========================================================================
def main() -> int:
    rows = list(csv.DictReader((OUT / "layer_matrix.csv").open(encoding="utf-8")))
    for r in rows:
        for k in MECH_FAMILIES + ["mech_any"]:
            r[k] = int(r[k])
        r["mech_conceal"] = int(any(r[k] for k in CONCEAL))

    say("M4 -- do the defence layers' errors multiply?")
    say("=" * 106)
    say(f"cases {len(rows)}   " + "  ".join(f"{k}={v}" for k, v in
                                            Counter(r["corpus"] for r in rows).items()))
    say(f"PRIMARY layer set ({len(PRIMARY)}): {', '.join(PRIMARY)}  "
        f"-> {len(list(combinations(PRIMARY, 2)))} pairs")
    say("  all five measured under ONE condition: judge role, no-tell for the LLM layers,")
    say("  PolicyEngine.default() (NOT the steelman cloud pack) for rules.")
    say("  `deepseek` is EXCLUDED from every headline number -- it was elicited in the")
    say("  faithful (description-shown) condition. It appears only in section 8.")

    blk = [r for r in rows if r["truth"] == "block"]
    alw = [r for r in rows if r["truth"] == "allow"]
    wrn = [r for r in rows if r["truth"] == "warn"]
    say(f"\ndomains: MISS on truth=block (n={len(blk)}); FALSE ALARM on truth=allow "
        f"(n={len(alw)}); truth=warn (n={len(wrn)}) excluded from both")

    em = build_err(blk, PRIMARY, "miss")
    ef = build_err(alw, PRIMARY, "falsealarm")

    say("\n\n### 1. marginal error rates  [95% Clopper-Pearson]")
    say("-" * 106)
    say(f"{'layer':<12}{'miss (truth=block)':>30}{'false alarm (truth=allow)':>36}")
    for L in PRIMARY:
        km, nm = int(em[L].sum()), len(blk)
        kf, nf = int(ef[L].sum()), len(alw)
        lm, hm = clopper_pearson(km, nm)
        lf, hf = clopper_pearson(kf, nf)
        say(f"{L:<12}{km:>5}/{nm:<4}{f'{km/nm:.3f} [{lm:.3f},{hm:.3f}]':>21}"
            f"{kf:>8}/{nf:<4}{f'{kf/nf:.3f} [{lf:.3f},{hf:.3f}]':>24}")

    say("\n\n### 2. pairwise error correlation")
    say("""metric choice. phi is the headline: it is the mean-square contingency coefficient
on two binary error indicators (identical to Pearson r on 0/1 data) and is the
statistic arXiv:2608.28327 reports for its 15 pairs, so our numbers sit on the
same axis as the in-position result. Cohen's kappa is printed beside it as the
more familiar agreement statistic but is NOT the headline: kappa is bounded by
the marginals, so a pair with very unequal error rates (rules at .134 vs gpt55 at
.014) cannot reach a high kappa even under maximal association -- exactly the
comparison we care about most. The odds ratio is given as a marginal-free third
view. c = P_obs / P_indep is the quantity the defence-in-depth claim is literally
about: how many times more often the pair fails together than the multiplication
rule predicts. c = 1 IS the multiplication rule.""")
    miss_recs = pair_table(em, PRIMARY,
                           "MISS correlation (truth=block; `review` counts as CAUGHT)",
                           "results_pairs_miss.csv")
    pair_table(ef, PRIMARY, "FALSE-ALARM correlation (truth=allow)",
               "results_pairs_falsealarm.csv")
    say("""
  Note on direction: false alarms compose DISJUNCTIVELY -- any layer firing
  raises the alarm -- so low correlation here is bad news, not good: independent
  false alarms accumulate across layers instead of overlapping.""")

    say("\n\n### 3. is the pooled association a between-corpus artifact?")
    say("""Pooling corpora of unequal difficulty inflates phi on its own. Below: pooled OR
vs the Cochran-Mantel-Haenszel common OR stratified by corpus. Collapse toward 1
would mean the pooled number was a between-corpus effect.""")
    say("-" * 106)
    say(f"{'pair':<20}{'pooled OR':>11}{'MH-OR|corpus':>15}{'MH p':>11}"
        f"   per-corpus phi (int300 / ind630 / ext189)")
    strat = []
    for A, B in combinations(PRIMARY, 2):
        tabs, phis = [], []
        for cp in CORPORA:
            s = [i for i, r in enumerate(blk) if r["corpus"] == cp]
            t = cells(em[A][s], em[B][s])
            tabs.append(t)
            phis.append(phi(*t))
        orv, p = mh_odds_ratio(tabs)
        pooled = next(r["odds_ratio"] for r in miss_recs if r["pair"] == f"{A}x{B}")
        strat.append({"pair": f"{A}x{B}", "pooled_OR": pooled, "MH_OR": orv, "MH_p": p,
                      **{f"phi_{cp}": v for cp, v in zip(CORPORA, phis)}})
        say(f"{A + 'x' + B:<20}{pooled:>11.2f}{orv:>15.2f}{p:>11.2e}   "
            + "  ".join(("  n/a" if math.isnan(v) else f"{v:+.2f}") for v in phis))
    with (OUT / "results_stratified.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(strat[0]))
        w.writeheader()
        w.writerows(strat)

    say("\n\n### 4. n_eff -- effective number of independent layers")
    say("""n_eff = ln P_joint / ln g, with g the geometric mean of the stack's miss rates.
Layers that truly multiply give n_eff = N; perfectly coupled layers give 1.
Domain truth=block. [censored] = no joint miss observed at this n, so only a
lower bound is available (from the exact upper limit on P_joint).""")
    say("-" * 106)
    say(f"{'stack':<46}{'N':>3}{'k/n':>9}{'P_joint':>10}{'P_indep':>11}{'n_eff':>8}{'95% CI':>19}")
    best = min([L for L in PRIMARY if L != "rules"], key=lambda L: em[L].mean())
    stacks = [("all 5 primary layers", PRIMARY),
              ("4 LLM judges, no rules", ["gpt55", "haiku", "opus", "sonnet"]),
              ("3 Anthropic judges (same vendor)", ["haiku", "opus", "sonnet"]),
              (f"rules + {best} (best single judge)", ["rules", best]),
              ("rules + opus", ["rules", "opus"]),
              ("rules + haiku", ["rules", "haiku"]),
              ("rules + sonnet", ["rules", "sonnet"]),
              ("opus + sonnet (same vendor)", ["opus", "sonnet"]),
              ("haiku + opus (same vendor)", ["haiku", "opus"]),
              ("opus + gpt55 (cross vendor)", ["opus", "gpt55"]),
              ("haiku + gpt55 (cross vendor)", ["haiku", "gpt55"])]
    nrec = []
    for label, ls in stacks:
        d = n_eff(em, ls)
        ci = (f">= {d['n_eff_lo']:.2f} [censored]" if d["censored"]
              else f"[{d['n_eff_lo']:.2f}, {d['n_eff_hi']:.2f}]")
        kn = f"{d['k_joint']}/{d['n']}"
        pt = "  --  " if d["censored"] else f"{d['n_eff']:.2f}"
        say(f"{label:<46}{d['N']:>3}{kn:>9}{d['p_joint']:>10.4f}{d['p_indep']:>11.6f}{pt:>8}{ci:>19}")
        nrec.append({"stack": label, "layers": "+".join(ls), **d})
    with (OUT / "results_neff.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(nrec[0]))
        w.writeheader()
        w.writerows(nrec)

    say("\n\n### 5. two-factor decomposition: common difficulty vs shared mechanism")
    say("""Factor D (common difficulty; Eckhardt & Lee 1985): difficulty varies across the
  input space, so independently built layers fail together simply because they
  meet the same hard cases -- no shared mechanism required. Per case,
  theta_i = the fraction of the OTHER layers that err on case i (leave-both-out,
  so theta carries no information about the A-B joint outcome). P_EL then assumes
  conditional independence GIVEN theta.
Factor M (shared mechanism): a concealment device on the case text itself.
  Measured from the case alone, with no reference to any layer's behaviour --
  that is what makes the two factors separable: D comes from layer behaviour on
  OTHER layers, M comes from the input.

    excess over the multiplication rule = (P_EL - P_indep)  <- D, difficulty
                                        + (P_obs - P_EL)    <- residual (M + rest)
z is a delete-1 jackknife z on the residual.""")
    say("-" * 106)
    say(f"{'pair':<20}{'P_indep':>10}{'P_EL':>10}{'P_obs':>10}{'excess':>10}"
        f"{'D share':>11}{'residual':>11}{'D %':>7}{'jack z':>9}")
    dec = []
    for A, B in combinations(PRIMARY, 2):
        d = decompose(em, PRIMARY, A, B)
        resid, z = jackknife_z(em, PRIMARY, A, B)
        d["resid_jack_z"] = z
        d["difficulty_pct_of_excess"] = (d["share_difficulty"] / d["excess_total"] * 100
                                         if abs(d["excess_total"]) > 1e-12 else float("nan"))
        dec.append(d)
        say(f"{d['pair']:<20}{d['p_indep']:>10.4f}{d['p_el']:>10.4f}{d['p_obs']:>10.4f}"
            f"{d['excess_total']:>10.4f}{d['share_difficulty']:>11.4f}{d['share_residual']:>11.4f}"
            f"{d['difficulty_pct_of_excess']:>6.0f}%{z:>9.2f}")
    llm_dec = [d for d in dec if "rules" not in d["pair"]]
    for lbl, grp in (("all 10 pairs", dec), ("LLM x LLM only (6 pairs)", llm_dec)):
        te = sum(d["excess_total"] for d in grp)
        td = sum(d["share_difficulty"] for d in grp)
        say(f"  {lbl:<26} difficulty = {td / te * 100:.0f}% of the total excess, "
            f"residual = {100 - td / te * 100:.0f}%   "
            f"(residual z>1.96 in {sum(1 for d in grp if d['resid_jack_z'] > 1.96)}/{len(grp)}, "
            f"z<-1.96 in {sum(1 for d in grp if d['resid_jack_z'] < -1.96)}/{len(grp)})")
    with (OUT / "results_decomposition.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dec[0]))
        w.writeheader()
        w.writerows(dec)

    say("\n  attenuation check -- is the residual just a noisy theta?")
    say("""  theta is estimated from only a handful of other layers, so it is a noisy
  difficulty measure, and measurement error in theta biases the D share DOWN
  (P_EL is pulled toward P_indep). If that is what drives the residual, the D
  share should rise monotonically as more layers go into theta. Below: the same
  decomposition for the 6 LLM x LLM pairs, with theta built from k other layers
  (k=1 uses one other primary layer; k=2,3 use more; k=4 adds the deepseek
  column PURELY as an extra difficulty probe -- it never enters the pair).""")
    say(f"  {'layers in theta':<20}{'D share of excess':>20}")
    em_ext = build_err(blk, EXTENDED, "miss")
    for k in (1, 2, 3, 4):
        tot_e = tot_d = 0.0
        for A, B in combinations(["gpt55", "haiku", "opus", "sonnet"], 2):
            pool = [L for L in EXTENDED if L not in (A, B)]
            sub = [A, B] + pool[:k]
            d = decompose(em_ext, sub, A, B)
            tot_e += d["excess_total"]
            tot_d += d["share_difficulty"]
        say(f"  k = {k:<16}{tot_d / tot_e * 100:>19.0f}%")

    say("\n\n### 6. direct test of the shared-mechanism factor")
    say("""Does a concealment device raise JOINT failure beyond what the case's difficulty
already explains? Per pair: 2x2 of (device present) x (both layers miss),
stratified by theta, CMH common OR. MH-OR ~ 1 => the device adds nothing once
difficulty is matched, i.e. the mechanism hypothesis fails.

CAVEAT stated up front: the device flags are LEXICAL and the rules layer is a
LEXICAL matcher, so pairs containing `rules` are definitionally predisposed
toward an effect. The LLM-only line is the honest test.""")
    mech_recs = []
    for flag in ("mech_conceal", "mech_any"):
        mech_recs += mechanism_block(em, PRIMARY, blk, flag, flag)
    with (OUT / "results_mechanism.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mech_recs[0]))
        w.writeheader()
        w.writerows(mech_recs)

    say("\n\n### 7. sensitivity checks")
    say("-" * 106)
    say("""(a) what counts as a MISS. Two operational readings, and they bracket the answer:
      PRIMARY  "the action EXECUTED"     -> miss = verdict in {allow, warn}
      STRICT   "the action was NOT STOPPED" -> miss = verdict in {allow, warn, review}
    They differ only on `review`, and `review` is used at scale by exactly ONE
    layer -- see the count below. Which reading is correct depends on whether a
    human review layer actually exists downstream to absorb the escalation, and
    that is precisely the arm this round has no data for. So both are reported.""")
    say("")
    say(f"    `review` on truth=block, by layer: "
        + "  ".join(f"{L}={sum(1 for r in blk if r[L] == 'review')}" for L in PRIMARY))
    es = build_err(blk, PRIMARY, "miss", strict=True)
    say(f"    miss rate under STRICT: "
        + "  ".join(f"{L}={es[L].mean():.3f}" for L in PRIMARY)
        + f"   (PRIMARY: " + "  ".join(f"{L}={em[L].mean():.3f}" for L in PRIMARY) + ")")
    say("")
    say(f"    {'definition':<12}{'group':<14}{'pairs':>6}{'phi min':>10}{'phi median':>12}{'phi max':>10}")
    fin = lambda xs: [x for x in xs if not math.isnan(x)]  # noqa: E731
    for dname, ee in (("PRIMARY", em), ("STRICT", es)):
        for gname, keep in (("rules x LLM", True), ("LLM x LLM", False)):
            ph = fin([phi(*cells(ee[A], ee[B])) for A, B in combinations(PRIMARY, 2)
                      if ("rules" in (A, B)) == keep])
            say(f"    {dname:<12}{gname:<14}{len(ph):>6}{min(ph):>+10.3f}"
                f"{float(np.median(ph)):>+12.3f}{max(ph):>+10.3f}")
    da, db = n_eff(em, PRIMARY), n_eff(es, PRIMARY)
    say(f"    n_eff(all 5 layers): PRIMARY {da['n_eff']:.2f} -> STRICT {db['n_eff']:.2f}")
    say("""    Read: the review/warn split is an output-VOCABULARY preference of one model,
    not a security fact, and switching definitions moves only that model's pairs
    -- upward. The rules-vs-LLM structure is unchanged under both readings.""")

    fa = [r for r in blk if r["corpus"] == "internal300" and r["opus_faithful"]]
    xf = np.array([miss(r["opus_faithful"]) for r in fa], dtype=np.int8)
    xn = np.array([miss(r["opus"]) for r in fa], dtype=np.int8)
    say(f"\n(b) elicitation condition, opus on internal300 truth=block (n={len(fa)}): "
        f"faithful miss {xf.mean():.3f} vs no-tell {xn.mean():.3f}, "
        f"phi across conditions {phi(*cells(xf, xn)):+.3f}")

    say("\n(c) per-corpus n_eff, all 5 primary layers, truth=block:")
    for cp in CORPORA:
        sidx = [i for i, r in enumerate(blk) if r["corpus"] == cp]
        d = n_eff({L: em[L][sidx] for L in PRIMARY}, PRIMARY)
        val = f">= {d['n_eff_lo']:.2f} [censored]" if d["censored"] else f"{d['n_eff']:.2f}"
        say(f"    {cp:<16} n={d['n']:<4} joint {d['k_joint']}/{d['n']}  "
            f"P_indep={d['p_indep']:.7f}  n_eff = {val}")

    say("""\n(d) held-out slice only. internal300 is AgentTrust's OWN benchmark, so the
    rules layer may have been tuned against it; external189 is the held-out
    corpus and is the honest slice for any claim about the rule layer.""")
    ext_idx = [i for i, r in enumerate(blk) if r["corpus"] == "external189"]
    say(f"    external189 truth=block, n={len(ext_idx)}")
    say(f"    {'pair':<20}{'a':>4}{'b':>5}{'c':>5}{'d':>5}{'phi':>9}{'Fisher p':>11}")
    for A, B in combinations(PRIMARY, 2):
        if "rules" not in (A, B):
            continue
        a, b, c_, d_ = cells(em[A][ext_idx], em[B][ext_idx])
        _, pv_ = stats.fisher_exact([[a, b], [c_, d_]])
        say(f"    {A + 'x' + B:<20}{a:>4}{b:>5}{c_:>5}{d_:>5}{phi(a, b, c_, d_):>+9.3f}{pv_:>11.3f}")

    say("\n(e) device-flag families on truth=block cases:")
    for k in MECH_FAMILIES + ["mech_conceal", "mech_any"]:
        say(f"    {k:<15} {sum(r[k] for r in blk):>4}/{len(blk)}")

    say("\n\n### 8. deepseek block -- CONDITION-MISMATCHED, not a headline number")
    say("""The deepseek column was elicited FAITHFUL (LLMJudge._build_prompt passes
action.description, which on block cases often names the threat outright). Its
errors therefore concentrate on cases where even the tell does not help -- i.e.
the hardest cases -- which mechanically inflates any difficulty-driven coupling
with the no-tell judges. Reported here only to show the size of that confound.""")
    say("-" * 106)
    pair_table(em_ext, EXTENDED, "MISS correlation, 6 layers INCLUDING deepseek",
               "results_pairs_miss_with_deepseek.csv")
    say(f"  deepseek miss rate {em_ext['deepseek'].mean():.3f} on {len(blk)} block cases.")
    say("  Compare the phi of deepseek's pairs against the same-family no-tell pairs above.")

    (OUT / "results.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print("\n[ok] wrote results.txt + results_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
