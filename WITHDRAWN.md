# Conclusions withdrawn during analysis

Seven candidate conclusions were stated during the analysis and withdrawn before the
paper was written. Each is listed with the check that removed it. The working notes
behind these entries are not shipped. The outputs in `analysis/M4/panel_v2b/` supersede
them. See also the audits summarised in `AUDITS.md`.

1. **"The layer that adds the most to the stack is the one with the worst solo
   accuracy."** Falsified by the second rule engine, b1, which has the worst solo accuracy
   of any layer and adds 9 cases. Source: the no-tell audit of b1, which needs the case
   texts and cannot be rerun from this repository.

2. **"Rule-plus-judge stacks beat judge-plus-judge stacks" as a headline.** In the
   stratified replication the held-out corpus favoured the rule stack in 0 of 12
   comparisons under PRIMARY (6 tied) and 1 of 12 under STRICT, and 4 of 12 and 9 of 12
   comparisons reversed direction between corpora. The comparison measures independence
   times competence, and the paper keeps the two apart. Source:
   `panel_v2b/results_stack_replication.txt`.

3. **"The rule layer and the LLM layer are independent."** Equivalence testing failed at
   both pre-registered bounds (OR within [0.5, 2] and |phi| < 0.30). The paper reports
   "no correlation we could detect", not independence. Source: `panel_v2b/results_tost.txt`,
   `PREREG_tost.md`.

4. **"Model version changes concentrate on the most sensitive batches."** Two events,
   one of them a counterexample; the densest segment of the external run was unaffected.
   On the primary panel the served-version change on the opus tier was associated with the
   corpus (chi-square by corpus), and no mechanism claim is made. Source: the
   per-call records of the two earlier re-collections for the two events (the second is
   `recollect_v2/calls_v2.jsonl`; the first is not part of this release);
   `panel_v2b/results_opus_by_version.txt` for the primary panel.

5. **"Cross-vendor judge pairs recover some independence."** On the primary panel the
   archived cross-vendor pairs' excess over the same-vendor pairs halves under STRICT and is
   confounded with that column's collection (different instruction header, batches of 90,
   no provenance). The cross-vendor pairs with clean provenance (the DeepSeek arm) read
   inside the same-vendor band. Source: `panel_v2b/results_neff.csv`,
   `panel_v2b/results_pairs_miss_with_deepseek.csv`.

6. **"The difficulty share has a lower bound near 47%."** The computation that produced
   it scored difficulty with a probe that admitted the archived faithful DeepSeek column, a
   column excluded from every other headline number, and the share moves by a factor of two
   with the probe. Withdrawn in favour of the range reported in the paper (31.8% to 61.8%
   across the four probes). Source: `analysis/M4/results_decomposition.csv` (archived panel, the k = 4 row) for the
   withdrawn number; `panel_v2b/results_extra_stats.txt`
   section 5 for the range now reported.

7. **"The re-collected judges saw no mention of this study."** A probe of the subagent
   system context showed a project-memory index naming the runtime and the venue. The claim
   is withdrawn; the affected collection (`recollect_v2/`) is retained as a comparison panel
   and the primary panel (`recollect_v2b/`) was collected in a session without that index.
   Source: a probe of the subagent system context, run before each re-collection. The probe
   transcripts are not part of this release.

# A corrected premise about prior work

This entry is kept separate because it concerns a statement about someone else's work,
not a result of ours.

**P1. "The effective-count measure n_eff was already used on judge panels by
arXiv:2605.29800."** Not the case. That paper's n_eff is the Kish effective sample size,
(sum w)^2 / sum w^2, a function of sample weights. Our quantity is ln P_joint / ln g, a
function of joint and marginal miss rates. Same name, different object. No number changed;
the measure was renamed n_mult (multiplication-equivalent layers) and is introduced in the
paper without a "following [X]" clause. A standing rule was adopted: every statement about
the literature is verified independently before it enters any deliverable, and the outcome
is recorded either way.
