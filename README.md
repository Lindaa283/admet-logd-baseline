# logD<sub>7.4</sub> baseline — a small, auditable molecular ML project

Predicting **logD<sub>7.4</sub>**, the octanol/water distribution coefficient at
pH 7.4, from 2D structure. One public dataset, one experimental endpoint, a
scaffold split, three zero-to-one-parameter baselines, five learned models, and
a protocol built so that "we did not tune on the test set" is something a
reviewer can *check* rather than something they have to take on trust.

The point of this repository is not the score. It is that every number in it
can be traced to the code and data that produced it, and that the places where
the method is weak are written down rather than left out.

> **What this project does not claim.** It establishes no link to dengue, to
> antiviral activity, to CYP3A4 inhibition, or to any biological endpoint.
> logD<sub>7.4</sub> is a physicochemical property. Its relationship to
> antiviral activity is neither tested nor assumed here. The repository was
> renamed from `admet-triage-dengue` precisely because that name implied a
> causal chain (ADMET model → antiviral triage) that this work does not support.

Chinese-language working documents: [`STATUS.md`](STATUS.md) (handoff),
[`results/experiment_log.md`](results/experiment_log.md) (every run, including
the failed ones), [`results/version_comparison.md`](results/version_comparison.md)
(comparison against a second, independent implementation in this same repo).

---

## 1. Status, stated plainly

| | |
|---|---|
| Data curated | ✅ 4200 compounds, sha256-pinned |
| Regression suite | ✅ 49 tests |
| Training, scaffold split | ✅ all 8 models |
| Training, random split (contrast) | ✅ all 8 models |
| Structure-only leakage audit | ✅ both splits |
| Model selection frozen | ✅ [`results/freeze_manifest.json`](results/freeze_manifest.json) |
| **Formal test-set evaluation** | ✅ **run once**, 2026-11-20 — see §7 |

Sections 5 and 6 report **validation-set** numbers. The test-set numbers appear
in §7 and nowhere else. The distinction is kept in the file names too:
`results/validation_*.json` drove model selection,
[`results/test_scaffold.json`](results/test_scaffold.json) holds the single
final evaluation, and
[`results/test_access_log.jsonl`](results/test_access_log.jsonl) has **exactly
one entry**.

### A scoping warning about this repository

This repository contains **two independent implementations**:

* **`src/logd/`** — this project, described by this README. Its test split was
  scored **once**, on 2026-11-20 (§7).
* **`src/admet/`** — a parallel implementation built concurrently by a
  different agent session, preserved untouched. It produced **its own,
  separate** test-set results, which live in `results/tables/` and
  `docs/ADMET_SESSION_HANDOFF.md`.

So "the test set in this repository was read once" is also **false**: it was
read once per implementation, by two pipelines that curate and split
differently. The correct statement is: *the `src/logd` test partition was
scored exactly once, and the `src/admet` test partition is a different set of
compounds scored by different code.* The two implementations
curate, split and evaluate differently, and
[`results/version_comparison.md`](results/version_comparison.md) establishes
which of their numbers are comparable at all (short answer: the scores are not
directly comparable; the split-difficulty measurements are, and they agree to
within 0.015 Tanimoto).

---

## 2. Reproduce it

```bash
pip install -r requirements.txt
export PYTHONPATH=src

python -m logd.data                      # download, curate, QC ledger
python -m pytest                         # 49 tests
python -m logd.train    --split scaffold  # CV on train, select on validation
python -m logd.train    --split random    # contrast split
python -m logd.audit                     # structure-only leakage audit
python -m logd.evaluate --split scaffold --confirm-final-evaluation
```

The last command is the only code path that scores the test set, it refuses to
run without the flag, and every invocation appends a record to
`results/test_access_log.jsonl`. If that log ever has more entries than there
are reported result sets, someone was peeking — and the log says so.

Thread caps used for the runs recorded here, because this machine
oversubscribes badly at 24 logical cores:

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
python -m logd.train --split scaffold --jobs 2
```

A pinned Linux container (`Dockerfile`, `run.ps1`) is provided as a
reproducibility aid. It is **not** required — see §9 for why it exists anyway.

**`artifacts/` and `data/` are not committed** (an 81 MB random forest and a
parquet file do not belong in git). Both are regenerable: the scaffold split is
deterministic with no RNG, the random split is seeded, and the frozen
estimators come from a fixed grid. `results/freeze_manifest.json` carries the
sha256 of every one of them, so a regenerated artifact can be checked
bit-for-bit against the run reported here rather than merely assumed to match.

**Test-suite caveat, stated because it is a real gap:** 49/49 passed after the
last change that the suite covers. The final two edits — making the headline
model pre-registered in `evaluate.py`, and correcting the `selection_rule`
string in `train.py` — were verified only by `py_compile` on all 12 modules and
by the 8 RDKit-free tests in `test_metrics.py`, because Smart App Control began
blocking RDKit before the full suite could be re-run (§9). `evaluate.py` has no
unit test in this suite either way; the first thing to do when RDKit imports
again is `python -m pytest`.

---

## 3. Data and curation

MoleculeNet *Lipophilicity*: 4200 experimental logD<sub>7.4</sub> values, an
AstraZeneca deposition distributed through ChEMBL, served by DeepChem.

* `raw_sha256 = aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd`,
  re-verified on every load.
* The exact ChEMBL deposition identifier is **deliberately not asserted**. An
  earlier version of this repository claimed `CHEMBL3301363`; that was never
  checked against a primary source, independent searching turned up DOI
  `10.6019/chembl3301361` instead, and neither could be confirmed. A provenance
  string that cannot be checked is worse than no string at all. See
  [`LICENSE`](LICENSE), which also states that the licence of the *data file* —
  as distinct from the code that serves it — could not be established.

Curation ledger ([`results/data_qc.json`](results/data_qc.json)), which is
forced to balance by an assertion in the code:

| | |
|---|---|
| raw rows | 4200 |
| unparseable SMILES | 0 |
| multi-fragment (counterion stripped) | 1 |
| missing label | 0 |
| duplicate canonical SMILES | 0 |
| **final rows** | **4200** |
| label range / mean / sd | −1.5 … 4.5 / 2.186 / 1.203 |

Three curation decisions that matter, all recorded in code comments so they
are not silently reverted:

1. **Charges are not neutralised.** logD<sub>7.4</sub> is a pH-dependent
   quantity. Rewriting the ionisation state would quietly change what the label
   means. Only counterions are stripped (largest fragment).
2. **Contradictory duplicate measurements are dropped, not averaged.**
   Averaging two measurements that disagree manufactures a label that was never
   measured. Tolerance 0.5 log units. (On this dataset: 0 such cases.)
3. **The `Ipc` descriptor is excluded.** It is an exponential of the adjacency
   matrix information content and overflows to `+inf` for mid-sized molecules.

### The duplicate count has a caveat, and it matters

"0 duplicate structures" is true **only at canonical-SMILES granularity**. At
the InChI connectivity layer — ignoring stereochemistry and protonation —
there are **82 groups covering 180 rows**. Because the Morgan fingerprint here
is built *without* chirality and all descriptors are 2D, those 82 groups are
**bit-identical in this project's feature space**. Collapsing them leaves
4200 − 180 + 82 = **4102** distinguishable feature vectors behind 4200 rows.

Under the scaffold split this is harmless: enantiomers share a Bemis–Murcko
framework, so they necessarily land in the same partition. Under the random
split it is **real leakage**, which is one reason the random split is reported
only as a contrast. §6 measures it.

---

## 4. Splits and protocol

| | |
|---|---|
| Headline split | Bemis–Murcko scaffold, 3360 / 420 / 420, deterministic (no RNG) |
| Contrast split | random, same sizes, `random_state = 20261120` |
| Hyperparameters | 5-fold `GridSearchCV` on **train only** |
| Model family choice | lowest **validation** RMSE, `argmin`, nothing else |
| Test set | scored **once**, on frozen `.joblib` files, with an access log |
| Metrics | RMSE, MAE, R², Spearman ρ, Kendall τ, each with a 95% percentile bootstrap interval (1000 resamples) |

`assert_no_leakage` runs on every split and checks for empty partitions,
molecule overlap, and — for the scaffold split — scaffold overlap. All
imputation, variance filtering and scaling live inside an sklearn `Pipeline`,
so they are fitted on training folds only and never see the fold they score.

### What the test partition was actually exposed to, precisely

Loose claims about test-set hygiene are the main thing this section exists to
avoid, including two of my own that I had to go back and fix:

* `train.py` **reads** the test rows and writes them to
  `artifacts/<split>/partition_test.csv`, to freeze which molecules are held
  out. It also passes them through `split_summary`, which records **five
  descriptive statistics of the test labels** — n, mean, sd, min, max — into
  `results/validation_<split>.json`. So those five numbers were computed before
  any test evaluation. No test row was featurised, fitted on, or scored, and
  the selection rule is an `argmin` over validation RMSE with no term that
  could read them. **The defensible claim is "no test-derived quantity entered
  any decision", not "the test set was untouched."** The string that
  `train.py` emitted originally ended with *"Test set untouched"*, which the
  same file contradicted three lines above; it is corrected in the code, and
  the two already-written JSONs carry a `_post_hoc_edit` field recording that
  exactly one string was rewritten and no number changed.
* `evaluate.py` originally reported whichever model scored **best on test** as
  `best_model`. That is post-hoc selection dressed up as a result, and it makes
  the bootstrap interval narrower than the truth. The headline model is now
  read back from `results/validation_<split>.json`, i.e. fixed before the test
  set is opened; if that pre-registration is missing, `evaluate.py` raises
  rather than falling back to a test-chosen winner. The post-hoc winner is
  still printed, labelled as not being the claim.

---

## 5. Validation results

**Measured, scaffold split, validation partition (n = 420).** Not test numbers.

| model | features | params | CV RMSE (train) | valid RMSE | valid MAE | valid R² | valid ρ | valid τ | fit (s) |
|---|---|---|---|---|---|---|---|---|---|
| `mean_baseline` | — | 0 | 1.197 | 1.214 | 0.971 | −0.006 | — | — | 2 |
| `clogp_identity` | Crippen clogP | **0** | 1.771 | 1.741 | 1.320 | −1.066 | 0.456 | 0.317 | 0.2 |
| `clogp_linear` | Crippen clogP | 2 | 1.098 | 1.057 | 0.859 | 0.239 | 0.456 | 0.317 | 0.04 |
| `ridge_ecfp4` | ECFP4 (2048) | — | 0.772 | 0.951 | 0.746 | 0.384 | 0.608 | 0.437 | 11 |
| **`svr_desc`** ★ | RDKit 2D desc. | — | **0.618** | **0.754** | **0.562** | **0.612** | **0.777** | **0.591** | 66 |
| `rf_desc` | RDKit 2D desc. | — | 0.666 | 0.819 | 0.638 | 0.543 | 0.698 | 0.515 | 586 |
| `hgb_desc` | RDKit 2D desc. | — | 0.635 | 0.795 | 0.608 | 0.569 | 0.728 | 0.544 | 753 |
| `hgb_desc_ecfp4` | desc. + ECFP4 | — | 0.613 | 0.789 | 0.606 | 0.576 | 0.732 | 0.547 | 1948 |

★ **pre-registered headline model**: `svr_desc` (RBF SVR, `C = 10`,
`epsilon = 0.1`, `gamma = scale`). Chosen on validation RMSE before the test
set was opened; pinned by hash in
[`results/freeze_manifest.json`](results/freeze_manifest.json).

Two things worth reading off that table:

* **The boosting models win on cross-validation and lose on validation.**
  `hgb_desc_ecfp4` has the best CV RMSE (0.613 vs 0.618) and a worse validation
  RMSE (0.789 vs 0.754). Cross-validation inside the training folds was not
  sufficient to pick the better-generalising family here — which is the whole
  argument for keeping an independent validation partition instead of selecting
  on CV alone.
* **The expensive models are not the good ones.** `hgb_desc_ecfp4` cost 32
  minutes to `svr_desc`'s 66 seconds and came out 0.035 log units worse.

**Random split (contrast), validation partition.** Reported to measure how much
the split protocol flatters a model, not as a result:

| model | valid RMSE | valid R² | valid τ |
|---|---|---|---|
| `mean_baseline` | 1.230 | −0.012 | — |
| `clogp_identity` | 1.745 | −1.034 | 0.288 |
| `clogp_linear` | 1.113 | 0.172 | 0.288 |
| `ridge_ecfp4` | 0.819 | 0.551 | 0.551 |
| `svr_desc` | 0.653 | 0.715 | 0.663 |
| `rf_desc` | 0.712 | 0.661 | 0.627 |
| `hgb_desc` | 0.677 | 0.694 | 0.637 |
| `hgb_desc_ecfp4` ★ | **0.645** | 0.722 | 0.660 |

★ `argmin` picked `hgb_desc_ecfp4` here, but it beats `svr_desc` by **0.008 log
units**. That is far below any reasonable uncertainty on 420 points; the honest
reading is "indistinguishable", not "boosting is better".

### How much the split protocol is worth

Same code, same data, same hyperparameter grid; only the split differs.

| | random | scaffold | cost of an honest split |
|---|---|---|---|
| `svr_desc` valid RMSE | 0.653 | 0.754 | **+0.101 log units** |
| `svr_desc` valid τ | 0.663 | 0.591 | −0.072 |
| `clogp_identity` valid τ | 0.288 | 0.317 | +0.029 |

The learned model loses 0.1 log units when the split stops being generous. The
zero-parameter clogP calculator is **unaffected** — it never learned anything
from the training set, so there is nothing for an easier split to flatter. That
contrast is the cleanest single argument in this repository for why a random
split on molecular data should not be believed.

---

## 6. Leakage audit (measured, not asserted)

A scaffold split guarantees distinct Bemis–Murcko frameworks. It does **not**
guarantee distant molecules, and it cannot catch two molecules that are
different strings but identical features. `logd.audit` measures both. It reads
structures only — no labels, no estimators, no predictions — so running it does
not consume the project's single permitted test evaluation, and it can be
re-run freely. Output:
[`results/leakage_audit_logd.json`](results/leakage_audit_logd.json).

| split | partition | nearest-train Tanimoto (median) | ≥ 0.7, close analogue | < 0.3, novel chemotype | **bit-identical ECFP4 twin in train** |
|---|---|---|---|---|---|
| scaffold | valid | 0.438 | 18.1% | 19.3% | **0** |
| scaffold | **test** | **0.457** | 16.9% | 14.8% | **0** |
| random | valid | 0.677 | 45.5% | 6.2% | 11 |
| random | **test** | **0.694** | 49.0% | 4.8% | **14** |

Read: the near-duplicate channel is **real and open in the random split** — 14
of 420 held-out molecules have a training molecule this project's features
cannot distinguish them from — and **closed in the headline scaffold split**.
The mechanism is the enantiomer/2D-descriptor identity described in §3, which
`assert_no_leakage` cannot catch because the canonical SMILES differ.

**Independent cross-check.** The parallel `src/admet` implementation measured
the same quantity with different curation, different fractions and separate
code, and got **0.442** (scaffold) and **0.677** (random) against this
project's 0.457 and 0.694 — agreement to within 0.015 Tanimoto. Its
connectivity-layer de-duplication also leaves only 2–6 identical twins in its
random split against this project's 14, which is direct quantitative evidence
that the de-duplication does what it claims.

---

## 7. Test-set result (scored once, on 2026-11-20)

The held-out scaffold-split test partition (**n = 420**) was scored **exactly
once**, against models frozen before the evaluation ran. The audit record is
[`results/test_access_log.jsonl`](results/test_access_log.jsonl):
`run_index: 1`, `2026-11-20T18:56:26Z`, git commit `d97df34`, bound to the
curated-data sha256. There is one entry and one reported result set.

| model | RMSE (95% CI) | MAE | R² | Spearman ρ | Kendall τ |
|---|---|---|---|---|---|
| `mean_baseline` | 1.240 [1.161, 1.325] | 1.032 | −0.006 | — | — |
| `clogp_identity` | 1.599 [1.480, 1.718] | 1.226 | −0.673 | 0.467 | 0.326 |
| `clogp_linear` *(best baseline)* | 1.099 [1.030, 1.167] | 0.919 | 0.209 | 0.467 | 0.326 |
| `ridge_ecfp4` | 0.972 [0.904, 1.039] | 0.766 | 0.382 | 0.602 | 0.429 |
| **`svr_desc` — pre-registered headline** | **0.809 [0.738, 0.881]** | **0.609** | **0.572** | **0.736** | **0.553** |
| `rf_desc` | 0.859 [0.792, 0.923] | 0.673 | 0.517 | 0.707 | 0.523 |
| `hgb_desc` | 0.839 [0.767, 0.913] | 0.645 | 0.539 | 0.722 | 0.536 |
| `hgb_desc_ecfp4` | 0.796 [0.727, 0.867] | 0.612 | 0.586 | 0.743 | 0.561 |

**The claim.** `svr_desc` (RBF SVR on 2D descriptors) improves RMSE by
**0.290 log units** over the strongest baseline, `clogp_linear`, and the
bootstrap intervals do not overlap ([0.738, 0.881] vs [1.030, 1.167]). That is
the result: a descriptor model beats a calibrated, zero-training-data cLogP
estimate on unseen chemotypes, by an amount larger than the uncertainty.

**What is not the claim.** `hgb_desc_ecfp4` has the lowest test RMSE (0.796),
0.013 log units below the headline, with intervals that almost completely
overlap. It is reported as `post_hoc_best_on_test` and is **not** this
project's result: it was identified after the test labels were seen, so its
interval understates its true uncertainty. The honest reading is that the two
models are indistinguishable, not that boosting won.

**Two baselines worth reading carefully.** Raw `clogp_identity` scores
R² = **−0.673**, i.e. worse than predicting the mean — because Crippen cLogP
estimates log*P*, not log*D* at pH 7.4, so it is systematically mis-calibrated
for this endpoint. Yet its τ = 0.326 equals the calibrated version's: the
*ranking* is fine and only the offset is broken. Reporting R² alone would have
hidden that; reporting τ alone would have hidden the calibration failure.

**Applicability domain — the advantage is real but local.** Test-to-train
nearest-neighbour ECFP4 Tanimoto: median 0.457, mean 0.497, range 0.216–0.949;
16.9% are close analogues (> 0.7) and 14.8% are novel chemotypes (< 0.3).
Error tracks that distance for the learned model and *not* for the baseline:

| nearest-train Tanimoto | n | `svr_desc` MAE | `clogp_linear` MAE |
|---|---|---|---|
| [0.0, 0.3) novel | 62 | 0.781 | 0.917 |
| [0.3, 0.5) | 167 | 0.681 | 0.909 |
| [0.5, 0.7) | 120 | 0.520 | 0.978 |
| [0.7, 1.0) analogue | 71 | 0.442 | 0.849 |

`svr_desc` degrades monotonically as compounds get further from the training
set (0.442 → 0.781 MAE), while `clogp_linear` stays flat — as it must, having
learned nothing from the training set. So the model's edge comes from
similarity to training chemistry and decays with novelty. On the most novel
sixth of the test set it is still better than the baseline, but by much less.

**Only the scaffold split was scored.** The random split stays
validation-only: it is a contrast, its leakage is already measured and known to
be non-zero, and scoring it would spend a second test evaluation to confirm
what §6 already shows.

Artifacts: [`results/test_scaffold.json`](results/test_scaffold.json),
[`results/test_table_scaffold.md`](results/test_table_scaffold.md),
`results/test_predictions_scaffold.csv`,
`results/figures/test_scaffold_svr_desc.png`.

### The state that was frozen before the evaluation ran

* [`results/freeze_manifest.json`](results/freeze_manifest.json) pins the git
  commit, every `config.py` constant, the raw and curated data hashes, the
  sha256 of all 12 `src/logd/*.py` files, the sha256 of all 8 frozen estimators
  and all 3 partition CSVs per split, every model's `best_params`, the metric
  list and the bootstrap settings. Anything that could change a reported test
  number also changes a hash in that file.
* The pre-registered headline model is `svr_desc` for the scaffold split.
* `results/test_access_log.jsonl` now holds exactly one entry. **A one-entry
  log does not by itself prove the set was read once** — a log can be
  truncated. What makes the claim checkable is that `evaluate.py` is the only
  module that reads `partition_test.csv` for scoring, that it contains no `fit`
  call, and that the frozen estimator hashes in the manifest can be recomputed
  against the artifacts on disk.

The evaluation had been blocked for several hours by Windows Smart App Control
(Enforced, `VerifiedAndReputablePolicyState = 1`) intermittently refusing
RDKit's unsigned native extension (§9). It was finally run behind a pre-flight
check that confirmed RDKit could load and featurise a molecule, specifically so
that a mid-run block could not leave an access-log entry with no result beside
it. It completed in a single pass.

---

## 8. Limitations

1. **One endpoint, one dataset, one assay family.** Nothing here generalises to
   another property, and nothing here says anything about biological activity.
2. **4200 molecules is small**, and only 4102 of them are distinguishable in
   this feature space (§3).
3. **One test evaluation means one number, not a distribution.** The test
   result (§7) comes from a single scaffold partition of 420 compounds. The
   bootstrap intervals quantify sampling noise *within* that partition; they do
   not capture how much the answer would move on a different scaffold
   partition. The validation-split comparison in §5 suggests that movement is
   comparable to the intervals themselves, so differences of less than about
   0.05 log units between models should not be read as real.
4. **14.8% of the scaffold test set is a novel chemotype** (Tanimoto < 0.3 to
   anything in training). Whatever the test RMSE turns out to be, it is an
   average over a set that is mostly interpolation; `evaluate.py` reports error
   banded by similarity for exactly this reason.
5. **2D features only, chirality discarded.** Enantiomers can have genuinely
   different logD; this model cannot express that in principle, not just in
   practice.
6. **No GPU model.** Deliberate, see §9.
7. **No applicability domain, no uncertainty per prediction.** The bootstrap
   intervals describe the metric, not individual predictions.
8. **The validation partition has been used once per model family**, which is
   what it is for; but the random-split "winner" differing from the
   scaffold-split winner by 0.008 log units is a reminder that selecting on 420
   points is itself noisy.
9. **One provenance fact and one licence fact could not be established** (§3).

---

## 9. Environment, cost, and why there is no GPU model

| | |
|---|---|
| Hardware | consumer laptop, 24 logical cores, 64 GB commit limit, RTX 5060 Laptop (8151 MiB) |
| GPU used | **no** |
| Monetary cost | **$0** — public data, local compute, no paid API |
| Wall clock, all 16 model fits | ≈ 109 min (56 min scaffold + 53 min random; `hgb_desc_ecfp4` alone is 32 + 30 min of that) |
| Python / key packages | 3.13 · rdkit 2026.09.2 · scikit-learn 1.9.1 · numpy 2.5.5 · pandas 3.1.1 · scipy 1.18.1 |

**No GPU model, on purpose.** With 4200 molecules and 216 RDKit 2D
descriptors (207 of which survive the variance filter), the descriptor models
finish in seconds to minutes on CPU — `svr_desc` in 66s.
Introducing a D-MPNN or ChemBERTa to justify owning a GPU would have cost a
PyTorch build for Blackwell `sm_120` (needs cu128+), hours of tuning, and a
real risk of ending up *worse* than a 66-second SVR, which is exactly what the
boosting models did. The honest version of "I have an RTX 5060" is "this
problem did not need it".

**Smart App Control is a real reproducibility hazard on this machine.** It
blocks RDKit's unsigned `.pyd` files with CodeIntegrity events 3033/3077/3118
under policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`. The block is
**intermittent**: the same file ran two full training jobs and a 49-test suite,
then was blocked again minutes later. Disabling Smart App Control is
irreversible without reinstalling Windows, so it was left alone. That is why
the pinned container exists — not as the primary path, but because the native
path on this machine cannot be relied on.

---

## 10. Next step, and the one fact that is already useful

The original motivation was to judge whether a model like this could be entered
into the Alchemistry competition. That judgement is **deferred** until the
official rules, data structures and evaluation metric are in hand. Nothing in
this project will be bent to fit a competition before then.

Three factual questions to answer when those materials arrive:

1. **What is the predicted endpoint?** If it is binding affinity or activity
   rather than a physicochemical property, this logD model **cannot** be
   reused — that is a different endpoint, not a tuning problem.
2. **What is the training data's structure?** Labelled? Congeneric series?
3. **Over how many compounds is Kendall τ computed?** What is the random
   fluctuation of τ at that n? `metrics.bootstrap_ci` answers precisely this.

**And one measured fact that is already relevant:** a **zero-parameter** Crippen
clogP calculator — no training, no fitting, shipped free inside RDKit — reaches
**τ = 0.317** on this endpoint's validation set and **τ = 0.326** on the
held-out test set (§7), *while scoring R² = −0.673 there*. So a target of
"τ ≥ 0.3" is not by itself an achievement, and a rank metric can look
respectable on a model whose absolute predictions are worse than the mean.
Before treating such a threshold as meaningful, measure what the free baseline
scores on *their* data, and ask which metric the ranking is being read from.

What transfers from this project is the **methodology**, not the weights:
scaffold splitting, leakage assertions, zero-parameter baselines, bootstrap
intervals, pre-registered model selection, and a test-access log.

---

## Licence

Code: see [`LICENSE`](LICENSE). It also records the data's upstream terms
(ChEMBL CC BY-SA 3.0, EMBL-EBI conditions, DeepChem MIT) and states explicitly
which two facts about the data file could not be verified.
