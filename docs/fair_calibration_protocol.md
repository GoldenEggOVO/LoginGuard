# Fair-calibration follow-up protocol

Recorded 2026-11-14, before evaluating the seeds and record range below. This follow-up addresses questions raised by already inspected results. It is a project-local frozen protocol, not an external preregistration.

## 1. Synthetic calibration control

Fit the existing seven-feature, 200-tree Isolation Forest on seed 42's first seven days. Choose its cutoff by the existing validation-F1 procedure on the next three days. Use the same validation records and labels to choose three rule thresholds from the Cartesian product:

- Account failures/60 seconds: 2, 3, 4, 5, 6, 8, 10.
- Distinct accounts/source/10 minutes: 2, 3, 4, 5, 6, 8, 10.
- Prior failures before a successful login: 1, 2, 3, 4, 5.

This is 245 rule candidates. Maximize validation event F1; ties use the first candidate in the listed enumeration. Retain the original fixed thresholds (5, 5, 3) as a separate reference. The forest's existing quantile grid has 80 positions before deduplication; the search spaces are disclosed, not claimed to have equal computational cost. The rule family uses fewer features than the forest, so this is a calibration control, not an algorithm-only comparison.

Lock the model, rule thresholds, validation metrics, and input/protocol/source hashes before generating evaluation streams. Evaluate generator seeds **311, 313, 317, 331, 337, 347, 349, 353**, absent from the archived seed tables at protocol creation. For each seed use two paired test profiles:

1. 20-second spacing, unfamiliar device, attack-specific source.
2. 150-second spacing, the target's primary device, shared source.

The first ten days provide each target stream's causal history, but do not refit or calibrate any detector. Evaluate the last four days only. Keep all generative labels, including early attempts. Report per-seed event counts, precision, recall, F1, FPR, and simulated-batch coverage, then unweighted means and ranges. These are new seeds within the same generator, not independent real-world evidence. The second profile changes three factors together and does not isolate their individual effects.

Pre-evaluation amendment, 2026-11-14: seed 307 was initially reserved but was inadvertently generated in a development test of selection-file timing. It is now a development fixture and is excluded from formal evaluation; seed 353 replaces it. This change was made before viewing formal holdout results. The threshold grid, methods, profiles, and public evaluation range are unchanged.

## 2. Public same-information controls

Local experiment records previously inspected the verified RBA archive through row 12,000,000. Reserve **rows 12,000,001-16,000,000** for this follow-up. Do not inspect their labels, scores, or summaries before locking the selection. Raw row ordinals define all ranges; reject equal timestamps at the three internal boundaries (training end, calibration end, evaluation start) before adaptation, rather than moving a boundary after seeing outcomes. Row 16M is a fixed processing cap; it is not claimed to finish a complete timestamp group or incident, and no later record is read to adjust it.

Use the same archive checksum and schema adaptation as the preceding studies. Preserve failed-status positives and the unresolved high-frequency pseudonym. All non-positive labels remain unlabelled. No raw identifiers are exported.

### Representations and methods

Use IP, ASN, country, OS, browser, and device type. Their global and personal probabilities, missing-value handling, and score-before-update order follow [Methods](METHODS.md). Keep personal alpha=10, selected using earlier development data; do not select alpha again.

For each successful login construct two vectors, each with six per-field surprisal values and six availability indicators. Missing fields contribute zero to their score position and zero to their indicator. A scalar control averages the six score positions over the available fields. Its forest counterpart receives exactly those score positions and indicators:

- Global mean surprisal versus forest on global per-field surprisal.
- Personal mean surprisal versus forest on personal per-field surprisal.

Include a forest on the five original status/count features as an additional reference. All three forests use 200 trees, random state 42, automatic subsampling, and one worker. **Fit each forest on successful records in rows 1-1,200,000 only.** This success-fitted count reference differs from the earlier all-record-trained forest; label it accordingly. Fit uses no labels. Context frequencies use observed successes, including unidentified attacks, and continue updating online throughout replay.

This comparison holds each context representation fixed across mean scoring and forest scoring. It does not compare arbitrary categorical algorithms or reproduce the full Freeman model.

### Calibration and score ties

Use successful records in rows 1,200,001-2,000,000 for every method. Predeclare calibration alert fractions **0.5%, 1%, and 2%**, with **2% primary**. These are fractions of all eligible calibration records, not true false-positive rates.

At each fraction b, admit exactly floor(b*n) calibration records. Rank by descending score, then ascending deterministic SplitMix64 key of the source row ordinal, with seed 20261114. Source ordinals are unique, and this 64-bit permutation supplies a reproducible, label-independent tie order. Save the boundary score and tie key, then apply that frozen rule to future records. This tie policy differs from the earlier conservative `nextafter` policy and must be reported explicitly. It does not guarantee equal future alert counts, operational capacity, or true FPR.

Write `selection.json` with all fitted-model hashes, representations, cutoffs, realized calibration counts, protocol hash, and source hashes **at the end of calibration and before reading the new evaluation range**. Records 2M-12M update histories only; they do not fit models, select methods, or change thresholds. Save job metadata and completed-run manifests separately from historical outputs.

### Reporting

Report all-record and successful-record denominators, known positives detected, unlabelled alerts, calibration counts, and no-history/with-history groups. The primary outcome is known-positive discovery and unlabelled volume at the frozen 2% calibration budget. Other budgets are prespecified sensitivity analyses, not a way to select a winner after evaluation. Known rows may be related; do not report public-data precision, F1, true FPR, significance, or production recall.

This follow-up does not change or retest the case-admission policy. Findings on the new range cannot turn earlier retrospective case analysis into blind validation. Further tuning informed by these results requires another uninspected range.

## Verification and provenance

Before evaluation, test validation-only rule selection, per-field/aggregate score equivalence, missingness, score-before-update, label isolation, exact calibration counts under ties, future-only tie decisions, and split guards. Check bounded whole/chunk replay equivalence and preservation of historical artifacts. Run a fresh-context review before reading the reserved records.

After evaluation, check cross-table totals, result hashes, source/model/protocol digests, and run completion. Record wall time, process peak memory, Python and package versions. These checks address implementation consistency and reproducibility, not independent confirmation of effectiveness. Keep personal learning documents, local plans, raw archives, and private repository history outside the public repository.
