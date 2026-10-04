#!/usr/bin/env python3
"""recollect_v2b -- per-case agreement between v2a (subagent system context WITH the project memory
index) and v2b (WITHOUT it) on one corpus, per tier, three codings. Numbers only, no interpretation.

Both panels: provenance_v2.jsonl (batched calls; first successful attempt = the main-analysis verdict).
Codings:  exact        4-value verdict identical
          PRIMARY-miss (verdict in {allow, warn}) identical           (review = caught)
          STRICT-miss  (verdict != block) identical                   (review = miss)
Subsets:  all                    every case with a verdict in both panels
          same_served_version    model_served_final identical in both panels (opus changes version)
          both_majority_version  neither record is tagged mixed-version (main-analysis rows of both)
Usage: python3 compare_v2a_v2b.py [--a DIR] [--b DIR] [--corpus external189]
Writes <b>/agreement_v2a_v2b_<corpus>.csv and prints the same table.
"""
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
M4 = HERE.parent
ap = argparse.ArgumentParser()
ap.add_argument("--a", default=str(M4 / "recollect_v2"))
ap.add_argument("--b", default=str(M4 / "recollect_v2b"))
ap.add_argument("--corpus", default="external189")
args = ap.parse_args()
TIERS = ["haiku", "sonnet", "opus"]


def load(d: str) -> dict:
    out = {}
    for l in (Path(d) / "provenance_v2.jsonl").open(encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r["corpus"] != args.corpus:
            continue
        assert (r["tier"], r["id"]) not in out, ("duplicate", r["tier"], r["id"], d)
        out[(r["tier"], r["id"])] = r
    return out


A, B = load(args.a), load(args.b)
prim = lambda v: v in ("allow", "warn")
strict = lambda v: v != "block"
rows = []
for tier in TIERS:
    keys = sorted(k for k in A if k[0] == tier and k in B)

    def stats(ks, label):
        n = len(ks)
        if n == 0:
            return {"corpus": args.corpus, "tier": tier, "subset": label, "n": 0, "exact": "", "primary_miss": "",
                    "strict_miss": "", "n_block_a": 0, "exact_on_block_a": "", "disagreements_a_to_b": ""}
        ex = sum(A[k]["verdict"] == B[k]["verdict"] for k in ks) / n
        pm = sum(prim(A[k]["verdict"]) == prim(B[k]["verdict"]) for k in ks) / n
        sm = sum(strict(A[k]["verdict"]) == strict(B[k]["verdict"]) for k in ks) / n
        blk = [k for k in ks if A[k]["verdict"] == "block"]
        exb = (sum(B[k]["verdict"] == "block" for k in blk) / len(blk)) if blk else ""
        dis = Counter(f"{A[k]['verdict']}→{B[k]['verdict']}" for k in ks if A[k]["verdict"] != B[k]["verdict"])
        return {"corpus": args.corpus, "tier": tier, "subset": label, "n": n, "exact": f"{ex:.3f}",
                "primary_miss": f"{pm:.3f}", "strict_miss": f"{sm:.3f}", "n_block_a": len(blk),
                "exact_on_block_a": (f"{exb:.3f}" if exb != "" else ""),
                "disagreements_a_to_b": "; ".join(f"{k}×{v}" for k, v in sorted(dis.items()))}

    rows.append(stats(keys, "all"))
    rows.append(stats([k for k in keys if A[k]["model_served_final"] == B[k]["model_served_final"]], "same_served_version"))
    rows.append(stats([k for k in keys if A[k].get("version_status") != "mixed-version"
                       and B[k].get("version_status") != "mixed-version"], "both_majority_version"))
    served = Counter((A[k]["model_served_final"].replace("claude-", ""), B[k]["model_served_final"].replace("claude-", "")) for k in keys)
    rows.append({"corpus": args.corpus, "tier": tier, "subset": "served_pairs_a|b", "n": len(keys), "exact": "", "primary_miss": "",
                 "strict_miss": "", "n_block_a": "", "exact_on_block_a": "",
                 "disagreements_a_to_b": "; ".join(f"{a}|{b}×{c}" for (a, b), c in sorted(served.items()))})

out = Path(args.b) / f"agreement_v2a_v2b_{args.corpus}.csv"
with out.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"[a] {args.a}\n[b] {args.b}\n[corpus] {args.corpus}  cases in a: {len({k[1] for k in A})}, in b: {len({k[1] for k in B})}")
print(f"{'tier':<7}{'subset':<24}{'n':>5}{'exact':>8}{'PRIMARY':>9}{'STRICT':>8}{'blk_a':>7}{'ex@blk':>8}  disagreements a→b")
for r in rows:
    print(f"{r['tier']:<7}{r['subset']:<24}{r['n']:>5}{r['exact']:>8}{r['primary_miss']:>9}{r['strict_miss']:>8}{r['n_block_a']:>7}{r['exact_on_block_a']:>8}  {r['disagreements_a_to_b']}")
print(f"[ok] wrote {out}")
