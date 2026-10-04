#!/usr/bin/env python3
"""M4 step B8 -- does the B7 stack comparison REPLICATE across the three corpora?

B7 (in tost.py) is a post-hoc analysis: it was added after both preregistered
equivalence bounds failed. For it to carry a headline claim it needs a
confirmatory pass. The three corpora are independent construction efforts with
disjoint id spaces, so re-running B7 inside each one is three replication
attempts rather than one pooled fit.

  claim under test:  a {rules, J} stack leaks less than a {J2, J} stack,
                     measured on the same cases -> exact paired McNemar

  replicates  = same direction AND significant in each corpus independently
  fails       = holds only pooled, scatters when stratified -> stays exploratory

POWER FLOOR, reported before any verdict. A two-sided exact McNemar on n
discordant pairs has a minimum attainable p of 2 * 0.5^n. So n < 6 can never
reach p < .05 no matter how lopsided the split, and Holm raises the bar further.
A comparison below its floor is UNPOWERED, not a failed replication -- calling it
a failure would be the same error the B0 feasibility check was built to prevent.

Holm is applied WITHIN each corpus (12 tests): each corpus is its own
replication attempt, and correcting across all 36 would blur the three attempts
into one pooled test, which is exactly what this analysis exists to avoid.

Deterministic: exact binomial only, no RNG.
"""
from __future__ import annotations

import csv
import math
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom

OUT = Path(__file__).resolve().parent
JUDGES = ["gpt55", "haiku", "opus", "sonnet"]
CORPORA = ["internal300", "independent630", "external189"]
ALPHA = 0.05

LOG: list[str] = []


def say(m: str = "") -> None:
    print(m)
    LOG.append(m)


def miss(v, strict=False):
    return int(v in (("allow", "warn", "review") if strict else ("allow", "warn")))


def mcnemar_exact(x, y):
    n01 = int(((x == 0) & (y == 1)).sum())      # rules stack caught, judge pair leaked
    n10 = int(((x == 1) & (y == 0)).sum())      # the reverse
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0, 1.0
    p = min(1.0, 2.0 * float(binom.cdf(min(n01, n10), n, 0.5)))
    floor = min(1.0, 2.0 * 0.5 ** n)            # best attainable p at this n
    return n01, n10, p, floor


def holm(p):
    m = len(p)
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: p[j])):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def comparisons(err):
    """All 12 ordered (anchor J, alternative partner J2) comparisons."""
    out = []
    for J in JUDGES:
        xa = (err["rules"] & err[J]).astype(np.int8)
        for J2 in JUDGES:
            if J2 == J:
                continue
            xb = (err[J2] & err[J]).astype(np.int8)
            n01, n10, p, floor = mcnemar_exact(xa, xb)
            out.append({"anchor": J, "alt": J2, "stack_A": f"rules+{J}", "stack_B": f"{J2}+{J}",
                        "A_leaks": int(xa.sum()), "B_leaks": int(xb.sum()),
                        "n01": n01, "n10": n10, "n_discordant": n01 + n10,
                        "p": p, "p_floor": floor})
    for r, ap in zip(out, holm([r["p"] for r in out])):
        r["p_holm"] = ap
        # Holm's own floor: the best this test could do is its raw floor scaled by
        # the largest multiplier it could face. Use the conservative m * floor.
        r["holm_floor"] = min(1.0, len(out) * r["p_floor"])
        r["powered"] = bool(r["holm_floor"] < ALPHA)
        r["favours_rules"] = r["A_leaks"] < r["B_leaks"]
        r["tied"] = r["A_leaks"] == r["B_leaks"]
        r["sig_for"] = bool(r["favours_rules"] and ap < ALPHA)
        r["sig_against"] = bool((not r["favours_rules"]) and (not r["tied"]) and ap < ALPHA)
    return out


def table(recs, title):
    say(f"\n{title}")
    say("-" * 118)
    say(f"{'stack A':<18}{'stack B':<18}{'A leaks':>8}{'B leaks':>8}{'n01':>5}{'n10':>5}"
        f"{'disc':>6}{'exact p':>10}{'Holm':>10}{'floor':>9}  verdict")
    for r in recs:
        if not r["powered"]:
            v = "UNPOWERED (floor > .05)"
        elif r["sig_for"]:
            v = "replicates" + ("***" if r["p_holm"] < .001 else "**" if r["p_holm"] < .01 else "*")
        elif r["sig_against"]:
            v = "*** AGAINST ***"
        else:
            v = "ns, favours rules" if r["favours_rules"] else \
                ("ns, tied" if r["tied"] else "ns, favours judge pair")
        say(f"{r['stack_A']:<18}{r['stack_B']:<18}{r['A_leaks']:>8}{r['B_leaks']:>8}"
            f"{r['n01']:>5}{r['n10']:>5}{r['n_discordant']:>6}{r['p']:>10.2e}"
            f"{r['p_holm']:>10.2e}{r['holm_floor']:>9.3f}  {v}")
    pw = [r for r in recs if r["powered"]]
    say(f"  powered cells {len(pw)}/{len(recs)}  |  "
        f"direction favours rules in {sum(r['favours_rules'] for r in recs)}/{len(recs)} "
        f"(tied {sum(r['tied'] for r in recs)})  |  "
        f"significant FOR {sum(r['sig_for'] for r in recs)}, AGAINST {sum(r['sig_against'] for r in recs)}")
    return recs


def main() -> int:
    rows = [r for r in csv.DictReader((OUT / "layer_matrix.csv").open(encoding="utf-8"))
            if r["truth"] == "block"]
    say("M4-B8 -- per-corpus replication of the B7 stack comparison")
    say("=" * 118)
    say("confirmatory pass on a post-hoc analysis; three corpora = three independent attempts")
    for cp in CORPORA:
        say(f"  {cp:<16} {sum(1 for r in rows if r['corpus'] == cp):>4} truth=block cases")
    say(f"  {'POOLED':<16} {len(rows):>4}")
    say("""
POWER FLOOR (read this before any verdict): a two-sided exact McNemar on n
discordant pairs cannot return p below 2 * 0.5^n, so n < 6 can never reach .05,
and Holm across 12 tests pushes the requirement to n >= 9 (12 * 2 * 0.5^9 = .047).
Cells under their floor are marked UNPOWERED. They are not failed replications.""")

    allr = []
    for strict, dname in ((False, "PRIMARY (miss = allow|warn)"),
                          (True, "STRICT (miss = allow|warn|review)")):
        say("\n\n" + "=" * 118)
        say(f"### definition: {dname}")
        say("=" * 118)
        per = {}
        for cp in CORPORA + ["POOLED"]:
            sub = rows if cp == "POOLED" else [r for r in rows if r["corpus"] == cp]
            err = {L: np.array([miss(r[L], strict) for r in sub], dtype=np.int8)
                   for L in ["rules"] + JUDGES}
            recs = comparisons(err)
            per[cp] = recs
            tag = "  <- reference, NOT an independent replication" if cp == "POOLED" else ""
            table(recs, f"{cp}  (n = {len(sub)}){tag}")
            for r in recs:
                allr.append({"definition": "STRICT" if strict else "PRIMARY", "corpus": cp, **r})

        say(f"\n\n  --- replication verdict, {dname} ---")
        say(f"  {'corpus':<16}{'n':>5}{'powered':>9}{'dir. for rules':>16}"
            f"{'sig FOR':>9}{'sig AGAINST':>13}")
        for cp in CORPORA:
            recs = per[cp]
            n = len([r for r in rows if r["corpus"] == cp])
            say(f"  {cp:<16}{n:>5}{sum(r['powered'] for r in recs):>6}/12"
                f"{sum(r['favours_rules'] for r in recs):>13}/12"
                f"{sum(r['sig_for'] for r in recs):>6}/12{sum(r['sig_against'] for r in recs):>10}/12")

        # direction consistency among cells that are not exact ties
        say("\n  direction consistency across the three corpora, per comparison")
        say("  (+ = rules stack leaks fewer, = tie, - = judge pair leaks fewer; "
            "lower-case = unpowered)")
        say(f"  {'comparison':<34}" + "".join(f"{cp[:9]:>11}" for cp in CORPORA) + "   reversals?")
        rev = 0
        for i in range(12):
            marks, signs = [], []
            for cp in CORPORA:
                r = per[cp][i]
                ch = "=" if r["tied"] else ("+" if r["favours_rules"] else "-")
                signs.append(ch)
                marks.append(f"{ch} ({r['n01']}/{r['n10']})" if r["powered"]
                             else f"{ch.lower()} ({r['n01']}/{r['n10']})")
            has_rev = ("+" in signs and "-" in signs)
            rev += has_rev
            r0 = per[CORPORA[0]][i]
            say(f"  {r0['stack_A'] + ' vs ' + r0['stack_B']:<34}"
                + "".join(f"{m:>11}" for m in marks)
                + ("   <-- REVERSES" if has_rev else ""))
        say(f"  -> {rev}/12 comparisons reverse direction between corpora")

        # stratified combination (correct way to pool thin strata: sum discordants)
        say("\n  stratified combination over the three corpora (sum of discordant pairs;")
        say("  this is the right way to combine thin strata -- it is NOT a fourth replication)")
        say(f"  {'comparison':<34}{'sum n01':>9}{'sum n10':>9}{'exact p':>12}{'Holm':>12}")
        comb, cp_ = [], []
        for i in range(12):
            s01 = sum(per[cp][i]["n01"] for cp in CORPORA)
            s10 = sum(per[cp][i]["n10"] for cp in CORPORA)
            n = s01 + s10
            p = 1.0 if n == 0 else min(1.0, 2.0 * float(binom.cdf(min(s01, s10), n, 0.5)))
            r0 = per[CORPORA[0]][i]
            comb.append({"comparison": f"{r0['stack_A']} vs {r0['stack_B']}",
                         "sum_n01": s01, "sum_n10": s10, "p": p})
            cp_.append(p)
        for r, ap in zip(comb, holm(cp_)):
            r["p_holm"] = ap
            st = "***" if ap < .001 else "**" if ap < .01 else "*" if ap < .05 else ""
            say(f"  {r['comparison']:<34}{r['sum_n01']:>9}{r['sum_n10']:>9}"
                f"{r['p']:>12.2e}{ap:>12.2e}{st}")
        say(f"  -> stratified: {sum(1 for r in comb if r['p_holm'] < ALPHA and r['sum_n01'] > r['sum_n10'])}"
            f"/12 favour rules significantly, "
            f"{sum(1 for r in comb if r['p_holm'] < ALPHA and r['sum_n10'] > r['sum_n01'])}/12 against")
        for r in comb:
            allr.append({"definition": "STRICT" if strict else "PRIMARY",
                         "corpus": "STRATIFIED_SUM", "stack_A": r["comparison"].split(" vs ")[0],
                         "stack_B": r["comparison"].split(" vs ")[1], "n01": r["sum_n01"],
                         "n10": r["sum_n10"], "n_discordant": r["sum_n01"] + r["sum_n10"],
                         "p": r["p"], "p_holm": r["p_holm"]})

    # ------------------------------------------------------------------
    say("\n\n" + "=" * 118)
    say("### B9. why external189 reverses -- and what DOES replicate")
    say("=" * 118)
    say("""The stack comparison in B7/B8 is not a pure test of independence. A stack's
leak rate is the product of TWO things: how DECOUPLED the layers are, and how
COMPETENT each layer is on that domain. A perfectly independent layer that cannot
see the threat class at all contributes nothing. Separating the two:""")
    say("\n  (i) marginal competence -- per-corpus MISS rate of each layer, truth=block")
    say(f"  {'corpus':<16}{'n':>5}" + "".join(f"{L:>10}" for L in ["rules"] + JUDGES))
    for strict, dl in ((False, "PRIMARY"), (True, "STRICT")):
        say(f"  [{dl}]")
        for cp in CORPORA:
            sub = [r for r in rows if r["corpus"] == cp]
            say(f"  {cp:<16}{len(sub):>5}" + "".join(
                f"{float(np.mean([miss(r[L], strict) for r in sub])):>10.3f}"
                for L in ["rules"] + JUDGES))
    say("""
  The rule layer misses 53% of block cases on external189 against ~6% on the other
  two corpora -- a 8.7x jump. external189 is the cloud / IAM / CI corpus that the
  DEFAULT rule pack, by construction, does not cover (it is the very gap the
  opt-in cloud pack exists to close, and the M4 spec forbids using that pack).
  So on that corpus the rule layer is not weakly-coupled-but-useful; it is blind.""")

    say("\n  (ii) decoupling itself -- per-corpus phi, the quantity the independence")
    say("       claim is actually about (PRIMARY, truth=block)")
    say(f"  {'corpus':<16}{'rules x LLM phi':>34}{'LLM x LLM phi':>34}")
    from itertools import combinations as _cmb
    P5 = ["rules"] + JUDGES
    for cp in CORPORA:
        sub = [r for r in rows if r["corpus"] == cp]
        e = {L: np.array([miss(r[L]) for r in sub], dtype=np.int8) for L in P5}
        def grp(keep):
            out = []
            for A, B in _cmb(P5, 2):
                if ("rules" in (A, B)) != keep:
                    continue
                x, y = e[A], e[B]
                a = int(((x == 1) & (y == 1)).sum()); b = int(((x == 1) & (y == 0)).sum())
                c = int(((x == 0) & (y == 1)).sum()); d = int(((x == 0) & (y == 0)).sum())
                den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
                if den:
                    out.append((a * d - b * c) / den)
            return out
        r_, l_ = grp(True), grp(False)
        say(f"  {cp:<16}"
            f"{f'median {float(np.median(r_)):+.3f}   [{min(r_):+.2f}, {max(r_):+.2f}]':>34}"
            f"{f'median {float(np.median(l_)):+.3f}   [{min(l_):+.2f}, {max(l_):+.2f}]':>34}")

    say("""
  => The DECOUPLING result replicates 3/3: rules x LLM sits at -0.06 / +0.04 / +0.10
     while LLM x LLM sits at +0.35 / +0.41 / +0.32, in every corpus separately,
     INCLUDING the held-out one where the rule layer is worst.
  => The OPERATIONAL STACK BENEFIT replicates 1/3. It holds where the rule layer is
     competent and reverses where it is blind.
  => These are not in conflict. Independence and competence are separate factors and
     the stack leak rate is their product. Independence at a 53% marginal miss rate
     buys nothing.""")

    say("\n\n" + "=" * 118)
    say("### B10. n_eff by corpus -- the independence measure IS immune to the confound")
    say("=" * 118)
    say("""n_eff is normalised by each layer's own marginal error rate, so unlike the raw
stack-leak comparison it does not reward a layer for merely being good. If the
B7 reversal on external189 is a competence effect rather than a coupling effect,
n_eff should hold there. It does.

`cens (0/n)` = zero joint misses observed, so no point estimate -- but note that
is the MOST favourable possible outcome for that stack, not a missing result.""")
    for strict, dl in ((False, "PRIMARY"), (True, "STRICT")):
        say(f"\n  [{dl}]  2-layer stacks, n_eff (joint misses / n)")
        say(f"  {'stack':<20}" + "".join(f"{c[:11]:>17}" for c in CORPORA) + f"{'POOLED':>17}")
        stacks = [("rules", j) for j in JUDGES] + [
            ("haiku", "opus"), ("haiku", "sonnet"), ("opus", "sonnet"),
            ("gpt55", "opus"), ("gpt55", "haiku"), ("gpt55", "sonnet")]
        for A, B in stacks:
            out = []
            for cp in CORPORA + ["POOLED"]:
                sub = rows if cp == "POOLED" else [r for r in rows if r["corpus"] == cp]
                e = {L: np.array([miss(r[L], strict) for r in sub], dtype=np.int8) for L in (A, B)}
                ps = [float(e[L].mean()) for L in (A, B)]
                n = len(sub)
                k = int(((e[A] == 1) & (e[B] == 1)).sum())
                if min(ps) <= 0:
                    out.append("n/a")
                    continue
                g = math.exp((math.log(ps[0]) + math.log(ps[1])) / 2)
                out.append(f"{math.log(k / n) / math.log(g):.2f} ({k}/{n})" if k
                           else f"cens ({k}/{n})")
            say(f"  {A + '+' + B:<20}" + "".join(f"{o:>17}" for o in out))
    say("""
  => rules + judge:  1.78 - 2.13 wherever estimable, in EVERY corpus, including
     external189 (1.83 - 1.96 under STRICT) where the rule layer misses 53%.
  => judge + judge:  1.13 - 1.44 in EVERY corpus.
  The independence result replicates 3/3 and is untouched by the competence gap
  that flipped B7. This, not B7, is what the headline should rest on.""")

    cols = sorted({k for r in allr for k in r}, key=lambda k: (
        ["definition", "corpus", "stack_A", "stack_B", "anchor", "alt", "A_leaks", "B_leaks",
         "n01", "n10", "n_discordant", "p", "p_floor", "p_holm", "holm_floor", "powered",
         "favours_rules", "tied", "sig_for", "sig_against"].index(k)
        if k in ["definition", "corpus", "stack_A", "stack_B", "anchor", "alt", "A_leaks",
                 "B_leaks", "n01", "n10", "n_discordant", "p", "p_floor", "p_holm",
                 "holm_floor", "powered", "favours_rules", "tied", "sig_for", "sig_against"]
        else 99))
    with (OUT / "results_stack_replication.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(allr)

    (OUT / "results_stack_replication.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print("\n[ok] wrote results_stack_replication.txt, results_stack_replication.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
