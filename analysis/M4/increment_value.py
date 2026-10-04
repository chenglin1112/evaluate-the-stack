#!/usr/bin/env python3
"""M4 追加 -- what does an INCREMENT actually buy a stack?

§9d showed a heterogeneous increment (the cloud rule pack) lifting the rule
layer's solo accuracy by 31% while adding exactly zero coverage to a
{rules, judge} stack. The obvious challenge is "you picked a weak pack". The
symmetric test settles it: hold the baseline stack fixed and compare two
candidate additions on the SAME cases.

  baseline stack  S = a set of judges (stack leaks iff EVERY member misses)
  addition A      either the remaining 4th judge (HOMOGENEOUS)
                  or the deterministic rule layer (HETEROGENEOUS)

  new coverage    leak(S) - leak(S + A)   = cases the stack leaked that A catches
  redundancy      of the block cases A catches, the share S already covered

One structural caveat, stated because it would otherwise be an unfair comparison:
§9d measured IMPROVING a layer already in the stack, whereas this measures ADDING
a layer. They are different operations. What makes them comparable is that both
reduce to the same operational question -- does the stack's joint-miss set
shrink -- and that question is asked here of one fixed baseline with two
candidate additions, so the two additions ARE directly comparable to each other.

Deterministic; exact binomial CIs only.
"""
from __future__ import annotations

import csv
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent
JUDGES = ["gpt55", "haiku", "opus", "sonnet"]
CORPORA = ["internal300", "independent630", "external189"]
LOG: list[str] = []


def say(m: str = "") -> None:
    print(m)
    LOG.append(m)


def miss(v, strict=False):
    return int(v in (("allow", "warn", "review") if strict else ("allow", "warn")))


def cp(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def stack_leak(err, layers):
    m = np.ones(len(err[layers[0]]), dtype=bool)
    for L in layers:
        m &= err[L] == 1
    return m


def main() -> int:
    rows = [r for r in csv.DictReader((OUT / "layer_matrix.csv").open(encoding="utf-8"))
            if r["truth"] == "block"]
    say("M4 追加 -- the value of an increment: homogeneous vs heterogeneous")
    say("=" * 116)
    say("""Same baseline stack, two candidate additions, same cases. `new coverage` is the
only number that matters operationally: how many of the cases the baseline stack
LEAKED does the addition catch. `redundancy` is the share of the addition's own
catches that the stack already had -- its denominator differs between the two
addition types (a rule pack catches few cases, a judge catches most), so compare
NEW COVERAGE across additions, and read redundancy only within an addition.""")

    recs = []
    for strict, dl in ((False, "PRIMARY"), (True, "STRICT")):
        for scope in ["POOLED"] + CORPORA:
            sub = rows if scope == "POOLED" else [r for r in rows if r["corpus"] == scope]
            err = {L: np.array([miss(r[L], strict) for r in sub], dtype=np.int8)
                   for L in ["rules"] + JUDGES}
            n = len(sub)
            say("\n" + "=" * 116)
            say(f"### {scope}   [{dl}]   truth=block, n={n}")
            say("=" * 116)

            for k_base in (2, 3):
                say(f"\n  baseline = {k_base} judges; additions compared on the identical baseline")
                say(f"  {'baseline stack':<26}{'leak':>6}  {'addition':<12}"
                    f"{'leak after':>11}{'NEW COVERAGE':>14}{'redundancy':>26}")
                for base in combinations(JUDGES, k_base):
                    lb = stack_leak(err, list(base))
                    nb = int(lb.sum())
                    adds = [(j, j) for j in JUDGES if j not in base] + [("rules", "rules")]
                    for name, layer in adds:
                        la = lb & (err[layer] == 1)
                        na = int(la.sum())
                        new = nb - na
                        catches = int((err[layer] == 0).sum())
                        already = int(((err[layer] == 0) & ~lb).sum())
                        lo, hi = cp(new, max(nb, 1))
                        kind = "HETERO" if layer == "rules" else "homo  "
                        red = f"{already}/{catches} = {already/catches:.1%}" if catches else "n/a"
                        say(f"  {'+'.join(base):<26}{nb:>6}  {name:<7}{kind:<5}"
                            f"{na:>11}{new:>14}"
                            f"{f'{red}':>26}")
                        recs.append({"definition": dl, "scope": scope, "n": n,
                                     "baseline": "+".join(base), "n_base": k_base,
                                     "addition": name,
                                     "kind": "heterogeneous" if layer == "rules" else "homogeneous",
                                     "leak_before": nb, "leak_after": na, "new_coverage": new,
                                     "new_cov_frac_of_leak": (new / nb) if nb else float("nan"),
                                     "addition_catches": catches, "already_covered": already,
                                     "redundancy": already / catches if catches else float("nan")})
                    say("")

    # ---- the headline contrast ------------------------------------------
    say("\n" + "=" * 116)
    say("### the contrast, summarised")
    say("=" * 116)
    for dl in ("PRIMARY", "STRICT"):
        for scope in ["POOLED", "external189"]:
            sel = [r for r in recs if r["definition"] == dl and r["scope"] == scope
                   and r["n_base"] == 3]
            homo = [r for r in sel if r["kind"] == "homogeneous"]
            het = [r for r in sel if r["kind"] == "heterogeneous"]
            say(f"\n  [{dl}] {scope}: baseline = 3 judges (4 such stacks)")
            say(f"    {'addition':<16}{'stacks':>7}{'leak before':>13}{'leak after':>12}"
                f"{'new coverage':>14}")
            for lbl, grp in (("+ 4th judge", homo), ("+ rule layer", het)):
                if not grp:
                    continue
                say(f"    {lbl:<16}{len(grp):>7}"
                    f"{'-'.join(str(r['leak_before']) for r in grp):>13}"
                    f"{'-'.join(str(r['leak_after']) for r in grp):>12}"
                    f"{'-'.join(str(r['new_coverage']) for r in grp):>14}")
            tot_h = sum(r["new_coverage"] for r in homo)
            tot_x = sum(r["new_coverage"] for r in het)
            say(f"    total new coverage over the 4 baselines:  "
                f"+4th judge = {tot_h}   |   +rule layer = {tot_x}")

    say("\n" + "=" * 116)
    say("### why PRIMARY is uninformative here, and STRICT is the reading")
    say("=" * 116)
    say("""Under PRIMARY the 3-judge baselines leak 1-4 cases in three of the four stacks --
no resolution at all. The one non-degenerate baseline (haiku+opus+sonnet, leak 19)
says a 4th judge buys 18 and the rule layer buys 17, i.e. a tie. That tie is an
artifact: decompose gpt55's 18 'new catches' by what it actually emitted.""")
    from collections import Counter as _C
    err_p = {L: np.array([miss(r[L]) for r in rows], dtype=np.int8) for L in ["rules"] + JUDGES}
    base3 = ["haiku", "opus", "sonnet"]
    lk = stack_leak(err_p, base3)
    newg = lk & (err_p["gpt55"] == 0)
    vb = _C(rows[i]["gpt55"] for i in np.where(newg)[0])
    say(f"\n  baseline haiku+opus+sonnet, PRIMARY: leak {int(lk.sum())}, "
        f"+gpt55 new coverage {int(newg.sum())}")
    say(f"  what gpt55 actually returned on those {int(newg.sum())} cases: {dict(vb)}")
    say(f"  -> {vb.get('review', 0)} of {int(newg.sum())} are `review`: DEFERRALS, not decisions.")
    err_s = {L: np.array([miss(r[L], True) for r in rows], dtype=np.int8) for L in ["rules"] + JUDGES}
    lk2 = stack_leak(err_s, base3)
    say(f"  Under STRICT the same baseline leaks {int(lk2.sum())} and +gpt55 buys "
        f"{int((lk2 & (err_s['gpt55'] == 0)).sum())} -- exactly the {vb.get('block', 0)} genuine "
        f"blocks, while +rules buys {int((lk2 & (err_s['rules'] == 0)).sum())}.")
    say("""  This is the THIRD place gpt55's review-vs-warn vocabulary flips a conclusion
  (§8a, §9b B7, here). The vocabulary difference is now itself a finding: a judge
  that defers instead of deciding scores well on every metric that credits
  deferral, and adds nothing to a stack with no human layer to absorb it.""")

    say("\n  redundancy, STRICT pooled (share of the addition's catches the stack already had):")
    say(f"  {'baseline':<28}{'addition':<10}{'redundancy':>16}{'new coverage':>14}")
    for b in combinations(JUDGES, 3):
        lkb = stack_leak(err_s, list(b))
        for add in [j for j in JUDGES if j not in b] + ["rules"]:
            cat = int((err_s[add] == 0).sum())
            alr = int(((err_s[add] == 0) & ~lkb).sum())
            say(f"  {'+'.join(b):<28}{add:<10}{f'{alr}/{cat} = {alr/cat:.2%}':>16}"
                f"{int((lkb & (err_s[add] == 0)).sum()):>14}")
    say("""  Note the redundancy percentages barely separate (96-99%) because their
  denominator is every case the layer catches (~500 of 560). NEW COVERAGE is the
  discriminating number; do not quote redundancy % as the contrast.""")

    say("""
  Read together with §9d:
    heterogeneous increment that IMPROVES a layer already in the stack
        (cloud pack on the rule layer): +31% solo accuracy, +0 stack coverage
    homogeneous increment that ADDS a layer (a 4th judge)          -> see above
    heterogeneous increment that ADDS a layer (the rule layer)     -> see above
  If both increments to a 3-judge stack buy little, the methodological claim is
  general. If the rule layer buys much more than a 4th judge, the claim is that
  redundancy tracks MECHANISM, not the mere act of adding.""")

    with (OUT / "results_increment.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    (OUT / "results_increment.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print("\n[ok] wrote results_increment.txt, results_increment.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
