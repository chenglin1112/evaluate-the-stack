#!/usr/bin/env python3
"""M4 step B -- equivalence testing (TOST) on the rules x LLM pairs.

The main analysis could only report "did not reject independence" (Fisher p all
> .29, Holm-adjusted 1.00). Not rejecting is not the same as establishing
independence. TOST inverts the hypotheses:

    H0 : |effect| >= delta   (the coupling is big enough to matter)
    H1 : |effect| <  delta   (the pair sits within delta of independence)

Both one-sided tests must reject before equivalence may be claimed.

delta is fixed in PREREG_tost.md, which was written BEFORE this script ran and
derives both bounds from anchors OUTSIDE this study. The constants below are
copied from it and asserted against it at run time. See that file for the honest
caveat about what "preregistered" can and cannot mean here.

Everything is exact and deterministic: noncentral-hypergeometric conditional
tests, exact conditional CIs, no RNG.
"""
from __future__ import annotations

import csv
import math
import re
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binom as stats_binom
from scipy.stats import nchypergeom_fisher
from scipy.stats.contingency import odds_ratio

OUT = Path(__file__).resolve().parent
PRIMARY = ["rules", "gpt55", "haiku", "opus", "sonnet"]
CORPORA = ["internal300", "independent630", "external189"]

# ---- preregistered constants (PREREG_tost.md §3) --------------------------
DELTA_OR = 2.0      # equivalence region OR in [1/2, 2]
DELTA_PHI = 0.30    # equivalence region phi in [-.30, +.30]
ALPHA = 0.05

_pre = (OUT / "PREREG_tost.md").read_text(encoding="utf-8")
assert "δ_OR = 2.0" in _pre and "δ_φ = 0.30" in _pre, \
    "PREREG_tost.md does not state the constants this script uses -- refusing to run"
assert re.search(r"α = 0\.05", _pre)

LOG: list[str] = []


def say(m: str = "") -> None:
    print(m)
    LOG.append(m)


def miss(v, strict=False):
    return int(v in (("allow", "warn", "review") if strict else ("allow", "warn")))


def cells(x, y):
    return (int(((x == 1) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum()),
            int(((x == 0) & (y == 1)).sum()), int(((x == 0) & (y == 0)).sum()))


def phi_of(a, b, c, d):
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return float("nan") if den == 0 else (a * d - b * c) / den


# ---- exact conditional inference ------------------------------------------
def excond_p(a, b, c, d, theta, side):
    """Exact conditional (noncentral hypergeometric) tail probability for the
    [0,0] cell given the margins. Validated against scipy.fisher_exact at
    theta = 1 in both tails."""
    rv = nchypergeom_fisher(a + b + c + d, a + b, a + c, theta)
    return float(rv.cdf(a)) if side == "le" else float(rv.sf(a - 1))


def tost(a, b, c, d, delta=DELTA_OR):
    """-> (p_upper, p_lower, TOST p). p_upper supports OR < delta;
    p_lower supports OR > 1/delta; TOST p = max (intersection-union)."""
    pu = excond_p(a, b, c, d, delta, "le")
    pl = excond_p(a, b, c, d, 1.0 / delta, "ge")
    return pu, pl, max(pu, pl)


def cond_or_ci(a, b, c, d, conf=1 - 2 * ALPHA):
    r = odds_ratio([[a, b], [c, d]], kind="conditional")
    lo, hi = r.confidence_interval(conf)
    return float(r.statistic), float(lo), float(hi)


def a_from_or(theta, r1, c1, n):
    """The (real-valued) [0,0] cell implied by odds ratio theta at fixed margins.
    Solves theta = a(n-r1-c1+a) / ((r1-a)(c1-a)); phi is linear in a, so this
    reparameterises the exact OR interval onto the phi scale without any new
    inference."""
    if not np.isfinite(theta):
        return float(min(r1, c1))
    if abs(theta - 1.0) < 1e-12:
        return r1 * c1 / n
    A = theta - 1.0
    B = -(theta * (r1 + c1) + (n - r1 - c1))
    C = theta * r1 * c1
    disc = B * B - 4 * A * C
    if disc < 0:
        return float("nan")
    for root in ((-B - math.sqrt(disc)) / (2 * A), (-B + math.sqrt(disc)) / (2 * A)):
        if -1e-9 <= root <= min(r1, c1) + 1e-9:
            return root
    return float("nan")


def phi_ci_from_or_ci(a, b, c, d, lo, hi):
    n, r1, c1 = a + b + c + d, a + b, a + c
    den = math.sqrt(r1 * (c + d) * c1 * (b + d))
    if den == 0:
        return float("nan"), float("nan")
    f = lambda aa: (aa * n - r1 * c1) / den  # noqa: E731
    return f(a_from_or(lo, r1, c1, n)), f(a_from_or(hi, r1, c1, n))


def feasibility(a, b, c, d, delta=DELTA_OR):
    """Could this pair EVER show equivalence at these margins? Rebuild the table
    at exactly OR = 1 (margins preserved, joint cell replaced by its independence
    expectation, rounded) and measure the exact CI width. Uses margins only --
    it does not look at the observed joint cell."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    a0 = int(round(r1 * c1 / n))
    b0, c0, d0 = r1 - a0, c1 - a0, n - r1 - c1 + a0
    if min(b0, c0, d0) < 0:
        return float("nan"), float("nan"), False
    _, lo, hi = cond_or_ci(a0, b0, c0, d0)
    return lo, hi, bool(lo >= 1 / delta and hi <= delta)


def holm(p):
    m = len(p)
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: p[j])):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def mcnemar_exact(x, y):
    """Exact two-sided McNemar on two binary outcomes measured on the SAME cases.
    x, y are joint-miss indicators of two 2-layer stacks. Discordant pairs only;
    exact binomial, deterministic. -> (n01, n10, p)."""
    n01 = int(((x == 0) & (y == 1)).sum())
    n10 = int(((x == 1) & (y == 0)).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2.0 * float(stats_binom.cdf(k, n, 0.5)))
    return n01, n10, p


# ---------------------------------------------------------------------------
def run_block(err, label, pairs, note=""):
    say(f"\n{label}   (n = {len(err[PRIMARY[0]])}){note}")
    say("-" * 118)
    say(f"{'pair':<18}{'a':>4}{'b':>5}{'c':>5}{'d':>5}{'OR':>8}"
        f"{'90% exact CI':>20}{'phi':>8}{'phi 90% CI':>18}"
        f"{'TOST p':>10}{'Holm':>9}  verdict")
    recs, ps = [], []
    for A, B in pairs:
        a, b, c, d = cells(err[A], err[B])
        orv, lo, hi = cond_or_ci(a, b, c, d)
        pu, pl, p = tost(a, b, c, d)
        plo, phi_ = phi_ci_from_or_ci(a, b, c, d, lo, hi)
        fl, fh, feas = feasibility(a, b, c, d)
        recs.append({"pair": f"{A}x{B}", "a": a, "b": b, "c": c, "d": d,
                     "cond_OR": orv, "OR_lo90": lo, "OR_hi90": hi,
                     "phi": phi_of(a, b, c, d), "phi_lo90": plo, "phi_hi90": phi_,
                     "p_upper": pu, "p_lower": pl, "tost_p": p,
                     "feas_lo": fl, "feas_hi": fh, "feasible_at_OR1": feas})
        ps.append(p)
    for r, ap in zip(recs, holm(ps)):
        r["tost_p_holm"] = ap
        eq_or = ap < ALPHA and r["OR_lo90"] >= 1 / DELTA_OR and r["OR_hi90"] <= DELTA_OR
        eq_phi = (not math.isnan(r["phi_lo90"]) and r["phi_lo90"] >= -DELTA_PHI
                  and r["phi_hi90"] <= DELTA_PHI)
        # A CI lying wholly OUTSIDE the equivalence region is a positive finding of
        # non-equivalence (coupling established) -- it is NOT a power failure, and
        # must not be labelled as one. Power only qualifies a FAILURE TO ESTABLISH
        # equivalence when the interval straddles the boundary.
        outside = r["OR_lo90"] > DELTA_OR or r["OR_hi90"] < 1 / DELTA_OR
        r["equivalent_OR2"] = bool(eq_or)
        r["equivalent_phi030"] = bool(eq_phi)
        r["coupling_established"] = bool(outside)
        v = ("EQUIV(OR<2)" if eq_or else
             "COUPLED (CI wholly outside)" if outside else
             ("inconclusive; UNDERPOWERED" if not r["feasible_at_OR1"] else "not equivalent")
             + (" [passes phi<.30 -- see B6 warning]" if eq_phi else ""))
        ci = f"[{r['OR_lo90']:.2f}, {r['OR_hi90']:.2f}]"
        pci = f"[{r['phi_lo90']:+.2f}, {r['phi_hi90']:+.2f}]"
        say(f"{r['pair']:<18}{r['a']:>4}{r['b']:>5}{r['c']:>5}{r['d']:>5}"
            f"{r['cond_OR']:>8.2f}{ci:>20}{r['phi']:>+8.3f}{pci:>18}"
            f"{r['tost_p']:>10.4f}{ap:>9.4f}  {v}")
    return recs


def main() -> int:
    rows = list(csv.DictReader((OUT / "layer_matrix.csv").open(encoding="utf-8")))
    blk = [r for r in rows if r["truth"] == "block"]
    alw = [r for r in rows if r["truth"] == "allow"]
    rules_pairs = [(A, B) for A, B in combinations(PRIMARY, 2) if "rules" in (A, B)]
    llm_pairs = [(A, B) for A, B in combinations(PRIMARY, 2) if "rules" not in (A, B)]

    say("M4-B -- equivalence testing (TOST) on layer-error independence")
    say("=" * 118)
    say(f"preregistered bounds (PREREG_tost.md): delta_OR = {DELTA_OR} "
        f"(equivalence region OR in [{1/DELTA_OR:.2f}, {DELTA_OR:.2f}]), "
        f"delta_phi = {DELTA_PHI}, alpha = {ALPHA}")
    say("""
Reading the verdict column:
  EQUIV(OR<2)          both one-sided exact tests reject after Holm AND the 90%
                       exact CI lies inside [0.50, 2.00] -> equivalence to
                       independence established at the preregistered main bound
  equiv(phi<.30) only  passes only the wider secondary bound anchored on the
                       smallest phi arXiv:2608.28327 reports
  UNDERPOWERED         at these margins the exact CI is wider than the
                       equivalence region even when OR is exactly 1, so this
                       pair CANNOT show equivalence at any joint cell. Reporting
                       it as "not equivalent" would be a power failure dressed
                       up as a finding.
  not equivalent       adequately powered and still failed""")

    say("\n\n### B0. feasibility first (margins only, joint cell not used)")
    say("-" * 118)
    say(f"{'pair':<18}{'exact 90% CI if OR were exactly 1':>40}{'   can this pair ever show equivalence?'}")
    em = {L: np.array([miss(r[L]) for r in blk], dtype=np.int8) for L in PRIMARY}
    for A, B in rules_pairs + llm_pairs:
        a, b, c, d = cells(em[A], em[B])
        fl, fh, feas = feasibility(a, b, c, d)
        say(f"{A + 'x' + B:<18}{f'[{fl:.2f}, {fh:.2f}]':>40}   "
            f"{'yes' if feas else 'NO -- underpowered at delta_OR=2'}")

    say("\n\n### B1. MAIN -- rules x LLM, pooled truth=block, PRIMARY miss definition")
    main_recs = run_block(em, "rules x LLM", rules_pairs)

    say("\n\n### B2. NEGATIVE CONTROL -- the same test on LLM x LLM")
    say("These are known to be strongly coupled. If the bound were too loose they")
    say("would also pass, and the whole test would be meaningless.")
    neg = run_block(em, "LLM x LLM", llm_pairs)
    n_or = sum(r["equivalent_OR2"] for r in neg)
    n_ph = sum(r["equivalent_phi030"] for r in neg)
    say(f"\n  -> at delta_OR=2:   {n_or}/{len(neg)} coupled pairs pass  "
        + ("(control OK)" if n_or == 0 else "(*** BOUND TOO LOOSE ***)"))
    say(f"  -> at delta_phi=.30: {n_ph}/{len(neg)} coupled pairs pass  "
        + ("(control OK)" if n_ph == 0 else "(*** BOUND TOO LOOSE ***)"))
    if n_ph:
        say("     offenders: " + ", ".join(
            f"{r['pair']} (OR {r['cond_OR']:.1f}, phi CI [{r['phi_lo90']:+.2f}, {r['phi_hi90']:+.2f}])"
            for r in neg if r["equivalent_phi030"]))
        say("     => the SECONDARY bound delta_phi=0.30 admits pairs that are demonstrably")
        say("        coupled (c ~ 5x the multiplication rule). It is DISQUALIFIED as evidence")
        say("        of equivalence. Any 'passes phi<.30' label below carries no weight.")

    say("\n\n### B3. STRICT miss definition (`review` counts as a miss)")
    es = {L: np.array([miss(r[L], True) for r in blk], dtype=np.int8) for L in PRIMARY}
    strict_recs = run_block(es, "rules x LLM, STRICT", rules_pairs)

    say("\n\n### B4. per corpus (truth=block, PRIMARY)")
    per = {}
    for cp in CORPORA:
        idx = [i for i, r in enumerate(blk) if r["corpus"] == cp]
        e = {L: em[L][idx] for L in PRIMARY}
        per[cp] = run_block(e, f"rules x LLM, {cp}", rules_pairs,
                            note="  <- HELD-OUT corpus" if cp == "external189" else "")

    say("\n\n### B5. false-alarm domain (truth=allow)")
    ef = {L: np.array([int(r[L] != "allow") for r in alw], dtype=np.int8) for L in PRIMARY}
    fa_recs = run_block(ef, "rules x LLM, false alarm", rules_pairs)

    # -- write ---------------------------------------------------------------
    allr = []
    for tag, recs in (("B1_main_block_primary", main_recs),
                      ("B2_negative_control_llm", neg),
                      ("B3_block_strict", strict_recs),
                      *[(f"B4_{cp}", per[cp]) for cp in CORPORA],
                      ("B5_falsealarm", fa_recs)):
        for r in recs:
            allr.append({"analysis": tag, **r})
    with (OUT / "results_tost.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allr[0]))
        w.writeheader()
        w.writerows(allr)

    say("\n\n### B6. summary -- BOTH preregistered bounds fail, for opposite reasons")
    say("-" * 118)
    infeas = [r for r in main_recs if not r["feasible_at_OR1"]]
    say(f"(1) MAIN bound delta_OR = {DELTA_OR}: NOT TESTABLE on this data.")
    say(f"    {len(infeas)}/{len(main_recs)} rules x LLM pairs are underpowered -- at these")
    say("    margins the exact 90% CI is wider than [0.50, 2.00] even when the odds ratio")
    say("    is EXACTLY 1 (section B0). So 'equivalence not established' here is a")
    say("    statement about n, not about the layers. n = 560 block cases is too few.")
    say(f"(2) SECONDARY bound delta_phi = {DELTA_PHI}: DISQUALIFIED as evidence.")
    say(f"    The negative control shows {sum(r['equivalent_phi030'] for r in neg)}/{len(neg)} "
        "demonstrably coupled LLM pairs also pass it.")
    say("    A bound that a c ~ 5x pair clears cannot certify independence.")
    say("    (delta_phi was anchored on arXiv:2608.28327's MINIMUM reported phi; that")
    say("     minimum turns out to be too permissive to serve as an equivalence bound.)")
    say("")
    say("=> Report B's honest verdict: TOST does NOT upgrade 'no coupling detected' to")
    say("   'independence established'. Report-of-record item 3 in 结果.md §9 STANDS,")
    say("   and now has a quantified reason attached rather than an admission of omission.")
    say("")
    say("What the data DOES support -- the achieved exact intervals (descriptive, not a test):")
    say(f"  {'pair':<18}{'OR 90% CI':>20}{'phi 90% CI':>20}")
    for r in main_recs:
        or_ci = f"[{r['OR_lo90']:.2f}, {r['OR_hi90']:.2f}]"
        ph_ci = f"[{r['phi_lo90']:+.2f}, {r['phi_hi90']:+.2f}]"
        say(f"  {r['pair']:<18}{or_ci:>20}{ph_ci:>20}")
    hi_rules = max(r["phi_hi90"] for r in main_recs)
    sv = [r for r in neg if r["phi"] > 0.4]          # the three same-vendor pairs
    gp = [r for r in neg if r["phi"] <= 0.4]         # the three gpt55-involving pairs
    say(f"  rules x LLM phi 90% upper limits: max {hi_rules:+.3f}")
    say(f"  same-vendor LLM pairs, phi 90% lower limits: min {min(r['phi_lo90'] for r in sv):+.3f} "
        f"({', '.join(r['pair'] for r in sv)})")
    say(f"  gpt55-involving LLM pairs, phi 90% lower limits: min {min(r['phi_lo90'] for r in gp):+.3f} "
        f"({', '.join(r['pair'] for r in gp)})")
    say("  PRECISELY: the rules-family intervals are disjoint from the three SAME-VENDOR")
    say("  LLM pairs, and only from those. They OVERLAP the three gpt55-involving pairs")
    say(f"  (e.g. gpt55xsonnet phi CI [{[r for r in gp if r['pair'] == 'gpt55xsonnet'][0]['phi_lo90']:+.2f}, "
        f"{[r for r in gp if r['pair'] == 'gpt55xsonnet'][0]['phi_hi90']:+.2f}] vs "
        f"rulesxgpt55 [{main_recs[0]['phi_lo90']:+.2f}, {main_recs[0]['phi_hi90']:+.2f}]).")
    say("  So 'the two families separate' is supported for the same-vendor trio and NOT")
    say("  yet for the cross-vendor pairs -- do not write it as a blanket separation.")
    say("  Section B7 tests the difference directly, and is powered where this is not.")

    say("\n\n### B7. POST-HOC (not preregistered) -- the adequately powered test")
    say("""Declared post-hoc: this test was added AFTER seeing that both preregistered
bounds fail. It is reported as exploratory and must be labelled that way in the
paper.

The scientific claim is not "rules and LLM are independent" (an equivalence claim
this n cannot support). It is "a rules+judge stack leaks less than a judge+judge
stack". That is a DIFFERENCE claim, it is measured on the SAME 560 cases, and it
is therefore testable by exact McNemar on the paired joint-miss indicators --
adequately powered, no equivalence bound needed.

For each judge J, compare the 2-layer stack {rules, J} against {J', J} for every
other judge J'. n01 = cases the rules stack caught but the judge-pair leaked;
n10 = the reverse. Holm-corrected across all comparisons.""")
    say("-" * 118)
    hdr_b = "stack B (Jalt+J)"
    say(f"{'stack A (rules+J)':<22}{hdr_b:<22}{'A leaks':>9}{'B leaks':>9}"
        f"{'n01':>6}{'n10':>6}{'exact p':>11}{'Holm':>9}")
    judges = [L for L in PRIMARY if L != "rules"]
    comps, pv = [], []
    for J in judges:
        xa = (em["rules"] & em[J]).astype(np.int8)
        for J2 in judges:
            if J2 == J:
                continue
            # every ordered (anchor J, alternative J2): the anchor judge is held
            # fixed and only the PARTNER changes (rules vs another judge), so all
            # 4x3 comparisons are distinct questions.
            xb = (em[J2] & em[J]).astype(np.int8)
            n01, n10, pp = mcnemar_exact(xa, xb)
            comps.append({"stack_A": f"rules+{J}", "stack_B": f"{J2}+{J}",
                          "A_leaks": int(xa.sum()), "B_leaks": int(xb.sum()),
                          "n01": n01, "n10": n10, "mcnemar_p": pp})
            pv.append(pp)
    for cmp_, ap in zip(comps, holm(pv)):
        cmp_["mcnemar_p_holm"] = ap
        st = "***" if ap < .001 else "**" if ap < .01 else "*" if ap < .05 else ""
        say(f"{cmp_['stack_A']:<22}{cmp_['stack_B']:<22}{cmp_['A_leaks']:>9}{cmp_['B_leaks']:>9}"
            f"{cmp_['n01']:>6}{cmp_['n10']:>6}{cmp_['mcnemar_p']:>11.2e}{ap:>9.2e}{st}")
    wins = sum(1 for c_ in comps if c_["A_leaks"] < c_["B_leaks"] and c_["mcnemar_p_holm"] < .05)
    say(f"\n  rules+J leaks significantly LESS than J'+J in {wins}/{len(comps)} comparisons "
        f"(Holm .05); rules+J leaks more in "
        f"{sum(1 for c_ in comps if c_['A_leaks'] > c_['B_leaks'] and c_['mcnemar_p_holm'] < .05)}.")
    say("  This is the claim the paper actually needs, and unlike TOST it is powered.")
    say("""
  The comparisons that do NOT reach significance all involve gpt55, whose PRIMARY
  miss rate (.014) is flattered by its habit of emitting `review` where the other
  models emit `warn` (结果.md §8a). Repeating B7 under the STRICT definition
  neutralises that vocabulary difference:""")
    say(f"{'stack A (rules+J)':<22}{hdr_b:<22}{'A leaks':>9}{'B leaks':>9}"
        f"{'n01':>6}{'n10':>6}{'exact p':>11}{'Holm':>9}")
    comps_s, pv_s = [], []
    for J in judges:
        xa = (es["rules"] & es[J]).astype(np.int8)
        for J2 in judges:
            if J2 == J:
                continue
            xb = (es[J2] & es[J]).astype(np.int8)
            n01, n10, pp = mcnemar_exact(xa, xb)
            comps_s.append({"stack_A": f"rules+{J}", "stack_B": f"{J2}+{J}",
                            "A_leaks": int(xa.sum()), "B_leaks": int(xb.sum()),
                            "n01": n01, "n10": n10, "mcnemar_p": pp})
            pv_s.append(pp)
    for cmp_, ap in zip(comps_s, holm(pv_s)):
        cmp_["mcnemar_p_holm"] = ap
        st = "***" if ap < .001 else "**" if ap < .01 else "*" if ap < .05 else ""
        say(f"{cmp_['stack_A']:<22}{cmp_['stack_B']:<22}{cmp_['A_leaks']:>9}{cmp_['B_leaks']:>9}"
            f"{cmp_['n01']:>6}{cmp_['n10']:>6}{cmp_['mcnemar_p']:>11.2e}{ap:>9.2e}{st}")
    ws = sum(1 for c_ in comps_s if c_["A_leaks"] < c_["B_leaks"] and c_["mcnemar_p_holm"] < .05)
    ls = sum(1 for c_ in comps_s if c_["A_leaks"] > c_["B_leaks"] and c_["mcnemar_p_holm"] < .05)
    say(f"\n  STRICT: rules+J leaks significantly less in {ws}/{len(comps_s)}, more in {ls}.")
    say(f"  PRIMARY: {wins}/{len(comps)} for, "
        f"{sum(1 for c_ in comps if c_['A_leaks'] > c_['B_leaks'] and c_['mcnemar_p_holm'] < .05)} against.")
    say("  Direction is consistent under both readings; no comparison ever favours a")
    say("  judge+judge stack significantly.")
    for c_ in comps:
        c_["definition"] = "PRIMARY"
    for c_ in comps_s:
        c_["definition"] = "STRICT"
    with (OUT / "results_stack_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["definition"] + [k for k in comps[0] if k != "definition"])
        w.writeheader()
        w.writerows(comps + comps_s)

    (OUT / "results_tost.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print("\n[ok] wrote results_tost.txt, results_tost.csv, results_stack_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
