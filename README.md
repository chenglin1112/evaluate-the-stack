# Evaluate the Stack, Not the Layer

Analysis code, and the data it runs on, for the paper

> Chenglin Yang. *Evaluate the Stack, Not the Layer: Do Deterministic and LLM Gates for Agent Actions Fail Independently?* arXiv preprint (identifier to be added), 2026.

Runtime gates for agent tool calls are usually stacked: a deterministic rule layer, then one or more LLM judges. The paper measures whether the errors of those layers are independent, on 1,119 labelled actions from three corpora, with one rule layer and four judges.

This repository holds the verdict of every layer on every case as a table, the scripts that turn that table into the statistics in the paper, and a record of which model version served each judge call.

## Rerunning the analysis

Python 3.9 with numpy 2.0.2, scipy 1.13.1 and statsmodels 0.14.6.

```
bash run_all.sh
shasum -a 256 -c EXPECTED_OUTPUTS.sha256
```

The first command takes about fifteen seconds and writes 57 output files next to the scripts. The second checks each of them against its expected hash. The only random step is the case-level bootstrap in `extra_stats.py`, which has a fixed seed (20260903, B = 4000).

## What is here

| path | content |
|---|---|
| `analysis/M4/panel_v2b/` | the primary panel: `layer_matrix.csv` and the scripts that analyse it |
| `analysis/M4/` | the same scripts on the archived panel (Section VI-A of the paper), with its own `layer_matrix.csv` |
| `analysis/M4/recollect_v2b/` | records of the re-collection behind the primary panel: one row per case (`provenance_v2.jsonl`), one per judge call (`calls_v2.jsonl`), the batched against single-case check, a summary of call counts, and `compare_v2a_v2b.py` |
| `analysis/M4/recollect_v2/` | the same two record files for the earlier re-collection, made with a project-memory index in the judges' context (Section VI-B) |
| `_recollect_deepseek.jsonl`, `_recollect_deepseek_faithful.jsonl` | the third-vendor judge: one record per API call, no-tell and faithful conditions |
| `PREREG_tost.md` | pre-registration of the equivalence test. It is in Chinese, and `tost.py` reads its constants from this file. `AUDITS.md`, section 5, summarises it |
| `AUDITS.md` | summary of the audits and of the deviations recorded during the re-collection |
| `WITHDRAWN.md` | the seven conclusions dropped during analysis, each with the check that removed it |
| `EXPECTED_OUTPUTS.sha256` | sha256 of every file that `run_all.sh` writes |

`M4` is the working name the study had during the analysis.

## The layer matrix

`layer_matrix.csv` has one row per case, 1,119 in all.

| column | values |
|---|---|
| `corpus` | `internal300`, `independent630`, `external189` |
| `id`, `category` | case identifier and threat or task category |
| `truth` | the label: `allow`, `warn` or `block` |
| `rules` | verdict of the deterministic rule layer: `allow`, `warn`, `block` or `review` |
| `gpt55`, `haiku`, `opus`, `sonnet` | verdicts of the four judges under the no-tell condition |
| `deepseek` | archived verdicts of a fifth judge under the faithful condition. Not used in any headline number |
| `opus_faithful` | the opus judge under the faithful condition, internal corpus only |
| `encoded`, `indirection`, `eval_exec`, `remote_exec`, `unicode_trick`, `mech_any` | 0/1 flags for concealment and delivery devices in the case text |

In `panel_v2b/` the `haiku`, `opus` and `sonnet` columns are the re-collected verdicts. In the top-level matrix they are the archived ones. The other columns are the same in both.

## Where each number comes from

File names are the outputs of `run_all.sh` in `analysis/M4/panel_v2b/`, unless they start with `../`.

| in the paper | file |
|---|---|
| marginal miss rates, pairwise phi, Fisher and Holm, CMH, difficulty share, mechanism test | `results.txt`, `results_pairs_miss.csv`, `results_pairs_falsealarm.csv`, `results_stratified.csv`, `results_decomposition.csv`, `results_mechanism.csv` |
| n_mult for every stack under both miss definitions | `results_neff.csv`, `results.txt` |
| equivalence test with the pre-registered bounds and the negative control | `results_tost.txt`, `results_tost.csv`, `PREREG_tost.md` |
| stratified replication, paired stack comparison | `results_stack_replication.txt`, `results_stack_replication.csv`, `results_stack_comparison.csv` |
| increments (Table III) | `results_increment.txt`, `results_increment.csv` |
| floors, Clopper-Pearson intervals, case-level bootstrap (Table V), per-corpus bands, Breslow-Day homogeneity, second implementation of the difficulty share | `results_extra_stats.txt`, `results_extra_stats_floors.csv`, `results_extra_stats_bootstrap.csv`, `results_extra_stats_percorpus.csv` |
| phi over phi_max | `results_phimax.csv` |
| served-model changes on the opus tier (Section VI-D) | `results_opus_by_version.txt`, `results_opus_by_version.csv` |
| archived against re-collected verdicts, third-vendor judge (Sections VI-A and VI-E) | `results_recollect.txt`, `results_recollect_pairs_primary.csv`, `results_recollect_pairs_strict.csv`, `layer_matrix_recollect.csv`, `results_pairs_miss_with_deepseek.csv` |
| memory panel (Section VI-B) | `../recollect_v2b/agreement_v2a_v2b_external189.csv` |
| batched against single-case judging (Section VI-C) | `../recollect_v2b/single_case_check_v2.csv` |
| call counts and tokens (LLM usage section) | `../recollect_v2b/calls_v2.jsonl`, `../recollect_v2/calls_v2.jsonl`, `../recollect_v2b/extract_summary_v2.json` |

## What is not here

The case texts, the prompts, the judges' raw outputs, the external corpus and the scripts that collected the verdicts are not in this repository. The cases are synthetic, and many of them are attack payloads.

Three sets of results in the paper need the case texts and cannot be recomputed from this repository: the cloud rule pack, the second rule engine b1, and the no-tell audit. The figure is not regenerated here either.

These materials are available from the author on request.

The internal and independent corpora and the rule layer belong to AgentTrust, https://github.com/chenglin1112/AgentTrust. The `rules` column is the verdict of its default rule set at commit `489d739b86afb917112d4f1149e691fcdaee0dd1`.

## The records

`provenance_v2.jsonl` has one record per case and judge tier. `calls_v2.jsonl` has one record per judge call: the model requested, the model that served the call, a refusal flag, the tools used, token counts, timestamps, and the sha256 of the prompt, of the instruction header and of the output. `agent_id` and `run_id` are opaque identifiers assigned by the harness that made the calls.

Free-text fields were removed from the records before release: the judges' reasons, the raw responses, and the text of refusals. Verdicts and every other field are unchanged. Of the 57 files that `run_all.sh` writes, 53 are byte for byte the files the numbers in the paper were taken from. The other four are verdict files derived from the third-vendor records, which differ only by the removed reason field.

## Corpora and labels

There are three corpora. The internal 300 is AgentTrust's own benchmark, six categories of fifty cases, and the set the rule layer was tuned on. The independent 630 was written by the AgentTrust project in five batches after a frozen release of the rule set. Cases that exposed a missing rule led to rule patches, so this corpus was built independently of the rules but is not zero-shot for the rule layer measured here. The external 189 was constructed with an LLM and labelled against a rubric. All 224 source cases were relabelled blind by an annotator who is not an author of the AgentTrust reports: 193 agreed and 31 disagreed (kappa 0.817 before filtering). The disagreements were dropped, and 189 cases remain after filtering for tells and leakage.

Labels: 560 block, 379 allow, 180 warn. Misses are counted on the 560 block cases and false alarms on the 379 allow cases. Warn cases are in neither. Two miss definitions are reported throughout: PRIMARY (miss = allow or warn) and STRICT (miss = allow, warn or review).

## Licence

Scripts are under the MIT Licence. Data files and documents are under CC BY 4.0. See `LICENSE`.

## Citing

```
@misc{yang2026stack,
  author = {Chenglin Yang},
  title  = {Evaluate the Stack, Not the Layer: Do Deterministic and {LLM} Gates for Agent Actions Fail Independently?},
  year   = {2026},
  note   = {arXiv preprint (identifier to be added)}
}
```

Questions: yangchenglin802@gmail.com, or open an issue.
