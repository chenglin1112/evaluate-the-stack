# Audits: English summary

This file summarises the audits carried out during the analysis and the deviations
recorded during the re-collection of the judge verdicts. The working notes behind it are
not part of this release.

## 1. Verification coverage

Question: which headline quantities have been recomputed by a second, independent
implementation?

Recomputed, with the reference used and the result:

| quantity | reference | result |
|---|---|---|
| rule-layer verdicts | `PolicyEngine.default()` re-run vs the frozen `rule_verdict` columns produced by another script at another time | 0 of 1,119 mismatches |
| Fisher exact test | `scipy.stats.fisher_exact` | library call, not re-implemented |
| exact conditional test inside TOST | own noncentral hypergeometric tail vs `scipy` at theta = 1 | agreement to 8 decimals, checked before the run |
| CMH common odds ratio | `statsmodels.StratifiedTable.oddsratio_pooled` | 10 of 10 pairs, relative difference 0 |
| phi | `scipy.stats.pearsonr` on 0/1 vectors | max difference 2.55e-15 |
| Cohen's kappa | independently written confusion-matrix formula | max difference 1.75e-15 |
| exact McNemar | `statsmodels` `mcnemar(exact=True)` | 12 of 12, difference 0 |
| n_mult point estimates | hand computation from the definition | difference 0 |
| Holm correction | `statsmodels.multipletests(method="holm")` | difference 0 |
| "cases added" counts | rewritten with set algebra on case ids | identical (rules 68, fourth judge 21 on the archived panel) |
| numbers typed into the draft | traced back to the source CSVs | 18 of 18; this check caught a PRIMARY/STRICT labelling error |
| determinism | full pipeline run twice | byte-identical outputs |

Not recomputed by a second implementation: the Eckhardt-Lee decomposition (the difficulty
share and its curve; no library counterpart exists), the interval endpoints propagated
into n_mult, and the delete-1 jackknife. The audit records that the CMH cross-check had
been promised during development and was only done at audit time. The paper reports the
difficulty share as a range across probes; a second implementation of the share, written
after review, agrees with the released script under the same probe construction (31.8%
against 32%) and is written by `panel_v2b/extra_stats.py` to `results_extra_stats.txt`.

## 2. Definition of n_mult

Two conclusions. First, reliability engineering has no standard quantity corresponding to
ours: its standard common-cause measure is the beta factor, a conditional probability, not
an effective count. Second, the paper that the internal specification had cited as a
precedent for "n_eff on judge panels" uses the Kish effective sample size, a function of
sample weights; same name, different quantity. Disposition: the measure was renamed n_mult
and is introduced as a new application of the effective-count construction (the lineage
runs through Hill numbers and effective numbers of parties and species), related to but
distinct from the beta factor and from Kish's n_eff. The normalisation by each layer's own
marginal rate is a design choice, stated in the paper, whose purpose is to separate
decoupling from competence. Numbers did not change.

## 3. Provenance of the archived judge outputs

Read-only audit of the fourteen archived judge-output files (not part of this release). Finding: the exact
model version, collection date, temperature and collection method of the four archived LLM
judges are not recorded anywhere in the workspace; the model identity is encoded only in
file names, and the union of JSON keys over all 4,776 rows is
`confidence, id, reason, risk_level, verdict`. Circumstantial evidence of manual collection
rather than API calls: the provider configuration template of the experiments directory
never listed keys for the vendors of those judges, and 289 of 630 rows of one file carry a
trailing-comma artefact consistent with copied output. The only archived column with API
provenance was the DeepSeek diagnostic column, which was excluded from the main analysis
for a different reason (elicitation conditions). Consequence adopted: the three re-collectable
tiers were re-collected with the served model recorded on every call (`recollect_v2b/`),
and the GPT column, which could not be re-collected, is labelled as archived wherever it
appears.

## 4. Corrected premises about prior work

Kept separate from the withdrawn conclusions because it concerns statements about other
people's work. One entry, P1, on the Kish n_eff (see `WITHDRAWN.md`). A standing rule was
adopted: every literature statement handed from planning to execution is verified before
it enters a deliverable, and the check is recorded whether it succeeds or fails.

## 5. Pre-registration of the equivalence test (`PREREG_tost.md`)

The document opens with a boundary statement: the main analysis had already been run when
the bounds were fixed, so the observed odds ratios of the four rules-by-judge pairs were
known. The document therefore does not claim a blind pre-registration. What it fixes is
weaker but checkable: the equivalence bounds are derived only from anchors outside this
study and are recorded before any TOST statistic was computed; `tost.py` copies the
constants from the document and asserts they match. One observed value was known to fall
outside the bound and the bound was kept.

Bounds: primary delta on the odds ratio of 2 (equivalence region OR in [0.5, 2]), anchored
in the engineering convention that a coupling factor changes a risk conclusion only when
it moves the joint probability by about an order of magnitude; secondary delta on phi of
0.30 (region [-0.30, 0.30]), anchored in the smallest inter-layer phi reported by the
seven-layer study, so that equivalence at this bound would mean coupling below anything
reported for a defence stack. The phi interval is a monotone transform of the exact
odds-ratio interval at fixed margins, not a second inference. Method: exact conditional
test on the two-by-two table, two one-sided tests, no randomness. Analysis sets declared in
advance: pooled block cases (n = 560) under PRIMARY, with STRICT, per-corpus and the
false-alarm domain reported alongside; negative control: the six judge-by-judge pairs are
run through the same procedure and are expected not to pass, and if they passed the bounds
would be judged too loose.

## 6. Deviations recorded during the re-collection

The deviations from the collection plan that are carried into the paper: the instruction header of the re-collections differs from the archived header
(one shared header, the example threat patterns dropped, a no-tools clause added; case
blocks byte-identical); batches of 10 in seeded random order instead of the archived
batches of 90; the opus tier was served by `claude-opus-4-8` on 50 of 112 batches while
`claude-opus-5` was requested, and the change was associated with the corpus; of the 50
batches, 12 followed a refusal event in the transcript, 2 show a version sequence without
a refusal, and 36 show no signal other than the per-call served-model field; the
pre-declared majority-version rule, which would have left 279 block rows with all five
layers, was set aside and the two served versions are reported side by side; one judge
emitted a tool call and three classifier warnings were raised, all kept. A probe of the judges' system context was run
before the collection (see `WITHDRAWN.md`, item 7).
