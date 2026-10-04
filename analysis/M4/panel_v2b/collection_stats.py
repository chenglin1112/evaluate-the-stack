#!/usr/bin/env python3
"""panel_v2b -- counts the paper reports about the collections themselves (numbers only).

  1. Section VI-A: the archived haiku / sonnet / opus columns against the re-collected ones.
     Exact verdict agreement; on the truth=block rows the miss rate of each column under both
     codings, the discordant counts and the exact McNemar test. Opus is compared on the cases
     whose batch was served by the tier's majority version.
  2. Section VI-C: batched against single-case judging.
  3. Section VI-D: served versions by batch, and the second opus version by corpus with the
     chi-square test of version by corpus.
  4. Appendix: false alarms shared with the rule layer, on the allow cases of the two corpora
     that the DeepSeek judge covers.
  5. LLM usage section: calls and tokens of the two re-collections.
miss and false alarm are analyze.py's own functions. Inputs: the two layer matrices,
../recollect_v2b/{provenance_v2.jsonl, calls_v2.jsonl, single_case_check_v2.csv} and
../recollect_v2/calls_v2.jsonl. Writes results_collection_stats.txt next to this file.
"""
from __future__ import annotations
import csv, json, sys
from collections import Counter
from pathlib import Path
from scipy import stats

OUT = Path(__file__).resolve().parent
M4 = OUT.parent
sys.path.insert(0, str(OUT))
import analyze as A  # helpers only; main() is guarded

TIERS = ["haiku", "sonnet", "opus"]
CORPORA = ["external189", "independent630", "internal300"]
LOG = []
def say(m=""):
    print(m); LOG.append(m)
def jl(p):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
def table(p):
    return list(csv.DictReader(p.open(encoding="utf-8")))

new = table(OUT / "layer_matrix.csv")
old = {r["id"]: r for r in table(M4 / "layer_matrix.csv")}
served = {(r["tier"], r["id"]): r["model_served_final"]
          for r in jl(M4 / "recollect_v2b" / "provenance_v2.jsonl") if r["kind"] == "batch"}
assert all((t, r["id"]) in served for t in TIERS for r in new), "every case needs a served version per tier"
majority = {t: Counter(v for (tt, _), v in served.items() if tt == t).most_common(1)[0][0] for t in TIERS}

say("panel_v2b -- collection counts (archived vs re-collected, single-case check, served versions, calls)")

say("\n### 1. archived against re-collected columns (Section VI-A)")
say("    cases = those whose batch was served by the tier's majority version; miss rates on truth=block rows,")
say("    archived vs re-collected, with discordant counts (archived only / re-collected only) and exact McNemar p")
for t in TIERS:
    rows = [r for r in new if served[(t, r["id"])] == majority[t]]
    k = sum(old[r["id"]][t] == r[t] for r in rows)
    blk = [r for r in rows if r["truth"] == "block"]
    parts = []
    for name, strict in (("PRIMARY", False), ("STRICT", True)):
        a = [A.miss(old[r["id"]][t], strict) for r in blk]
        b = [A.miss(r[t], strict) for r in blk]
        d10 = sum(1 for x, y in zip(a, b) if x and not y)
        d01 = sum(1 for x, y in zip(a, b) if y and not x)
        p = stats.binomtest(min(d10, d01), d10 + d01, 0.5).pvalue if d10 + d01 else 1.0
        parts.append(f"{name} {sum(a) / len(blk):.3f} vs {sum(b) / len(blk):.3f} ({d10}/{d01}, p={p:.2g})")
    say(f"  {t:<7} agreement {k}/{len(rows)} = {k / len(rows):.3f}   block rows n={len(blk)}: " + "   ".join(parts))

say("\n### 2. batched against single-case judging (Section VI-C)")
single = table(M4 / "recollect_v2b" / "single_case_check_v2.csv")
say(f"  {'tier':<7}{'n':>5}{'exact':>8}{'PRIMARY miss':>14}{'STRICT miss':>13}")
for t in TIERS:
    s = [r for r in single if r["tier"] == t]
    rate = lambda col: sum(int(r[col]) for r in s) / len(s)
    say(f"  {t:<7}{len(s):>5}{rate('agree_exact'):>8.3f}{rate('agree_primary_miss'):>14.3f}{rate('agree_strict_miss'):>13.3f}")

say("\n### 3. served versions (Section VI-D)")
calls = jl(M4 / "recollect_v2b" / "calls_v2.jsonl")
for t in TIERS:
    b = [c for c in calls if c["tier"] == t and c["kind"] == "batch"]
    say(f"  {t:<7} {len(b)} batches: " + ", ".join(f"{m} {n}" for m, n in sorted(Counter(c["model_served_final"] for c in b).items())))
ob = [c for c in calls if c["tier"] == "opus" and c["kind"] == "batch" and c["model_served_final"] != majority["opus"]]
ref = sum(1 for c in ob if c["refusal_seen"])
seq = sum(1 for c in ob if not c["refusal_seen"] and len(set(c["models_served_sequence"])) > 1)
say(f"  opus batches not served by {majority['opus']}: {len(ob)} = {ref} after a refusal event"
    f" + {seq} with a version sequence and no refusal + {len(ob) - ref - seq} with neither")
say("  cases whose opus verdict came from the second version, by corpus:")
for label, rows in (("truth=block rows", [r for r in new if r["truth"] == "block"]), ("all cases", new)):
    tab, cells = [], []
    for c in CORPORA:
        sub = [r for r in rows if r["corpus"] == c]
        k = sum(1 for r in sub if served[("opus", r["id"])] != majority["opus"])
        tab.append([k, len(sub) - k]); cells.append(f"{c} {k}/{len(sub)} ({k / len(sub):.0%})")
    chi2, p, dof, _ = stats.chi2_contingency(tab, correction=False)
    say(f"    {label:<17} " + ", ".join(cells) + f"; chi-square({dof}) = {chi2:.1f}, p = {p:.1e}")

say("\n### 4. false alarms shared with the rule layer (appendix)")
ben = [r for r in new if r["truth"] == "allow" and r["corpus"] in ("external189", "independent630")]
say(f"  allow cases of the external and independent corpora: n = {len(ben)}; rules false alarms {sum(A.false_alarm(r['rules']) for r in ben)}")
for col in ("deepseek", "gpt55", "haiku", "opus", "sonnet"):
    say(f"  {col:<9} false alarms {sum(A.false_alarm(r[col]) for r in ben):>3}, shared with rules "
        f"{sum(1 for r in ben if A.false_alarm(r['rules']) and A.false_alarm(r[col]))}")
say("  (deepseek here is the archived column; the re-collected one is in results_recollect.txt)")

say("\n### 5. calls and tokens (LLM usage section)")
for name, d in (("primary re-collection (recollect_v2b)", "recollect_v2b"),
                ("comparison re-collection (recollect_v2)", "recollect_v2")):
    cs = jl(M4 / d / "calls_v2.jsonl")
    kinds = Counter(c["kind"] for c in cs)
    say(f"  {name}: {len(cs)} calls, {kinds['batch']} batched and {kinds['single']} single-case;"
        f" second attempts {sum(1 for c in cs if c['attempt'] > 1)}; calls with no output {sum(1 for c in cs if not c['n_messages'])}")
    say(f"    input tokens {sum(c['tokens_input_total'] or 0 for c in cs):,}; output tokens {sum(c['tokens_output_total'] or 0 for c in cs):,};"
        f" refusal events {sum(1 for c in cs if c['refusal_seen'])}; calls that used a tool {sum(1 for c in cs if c['tools_used'])}")

(OUT / "results_collection_stats.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
print("\n[ok] wrote results_collection_stats.txt")
