# Experiments

LoginGuard examines how calibration, input representation, behavior changes, and investigation capacity affect authentication anomaly detection. All scoring uses causal log history. See [Methods](METHODS.md) for definitions and [Data](DATA.md) for source limits and split boundaries.

## Fair rule calibration

The original seed-42 comparison detected 17/86 attack-labelled events with fixed rules and 84/86 with a validation-calibrated forest. A follow-up gives the rule family access to the same validation labels, selecting thresholds (2, 3, 5) from 245 candidates. The forest retains its validation cutoff 0.6882307636065044. Both are frozen before eight new generator seeds; target prefixes supply history without refitting.

Values below are unweighted means across seeds. The forest uses seven features; the rule family uses fewer, so this controls calibration access rather than every information/capacity difference.

| Profile | Method | Event recall | F1 | Normal-event FPR | Batch coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| Unfamiliar device, 20s, attack-specific IP | Fixed rules | 0.086 | 0.144 | 0.0133 | 0.328 |
| Same | Calibrated rules | 0.763 | 0.822 | 0.0127 | 1.000 |
| Same | Forest | 0.972 | 0.939 | 0.0135 | 1.000 |
| Primary device, 150s, shared IP | Fixed rules | 0.101 | 0.166 | 0.0141 | 0.367 |
| Same | Calibrated rules | 0.187 | 0.286 | 0.0159 | 0.422 |
| Same | Forest | 0.631 | 0.724 | 0.0153 | 0.898 |

Calibration narrows the fast-profile gap; both calibrated methods lose detections under the combined stress profile. The latter changes three factors together and cannot attribute the change to one factor. Seeds vary one simulator, not a real population. Batch coverage means at least one alerted member, not correct incident attribution.

[Summary](../reports/fair_calibration/summary.csv) · [Per-seed counts](../reports/fair_calibration/per_seed.csv) · [Validation candidates](../reports/fair_calibration/rule_validation_candidates.csv) · [Selection](../reports/fair_calibration/selection.json)

## Same-information public controls

The latest evaluation uses RBA rows 12,000,001-16,000,000: 4M records, 1,789,973 successes, and 30 known positives, all successful. Three forests fit the same 573,654 successful training records. All methods calibrate on the same 380,910 successes; the prespecified 2% primary budget admits exactly 7,618 calibration records using the recorded tie policy.

Global and personal representations each contain six per-field surprisals and six availability indicators. Within each representation, the mean scorer and forest use the same evidence. Alpha=10 is retained from earlier development selection, which used labels. No method or threshold was selected using this new range.

| Method | Known-positive records detected | Unlabelled alerts |
| --- | ---: | ---: |
| Success-fitted count forest | 0/30 | 41,746 |
| Global field forest | 27/30 | 61,059 |
| Global mean | 29/30 | 59,632 |
| Personal field forest | 29/30 | 47,309 |
| Personal mean | 29/30 | 42,630 |

Context representations recover known positives missed by the count representation. Mean scoring has a better primary outcome than the corresponding forest here; this does not establish a general algorithm ranking. Equal discovery counts do not prove identical alert sets. Personalization reduces aggregate unlabelled volume but raises the unlabelled-success alert rate for records without prior success: 7.701%, versus 4.762% for global mean.

Budgets of 0.5% and 1% were also specified before evaluation. Personal mean detects 29/30 at both, with 13,375 and 23,971 unlabelled alerts respectively. Those sensitivity results do not replace the 2% primary comparison or select a production threshold.

[All budgets and history groups](../reports/rba_fair/metrics.csv) · [Selection](../reports/rba_fair/selection.json) · [Run measurements](../reports/rba_fair/audit.json) · [Recorded protocol](fair_calibration_protocol.md)

## Behavior, memory, and workload

| Question | Observation within the specified experiment | Results |
| --- | --- | --- |
| Device familiarity | At 20s and a fixed attack-specific source, mean forest recall changed from 0.971 to 0.758 when using the primary device | [Factorial study](../reports/mimicry/summary.csv) |
| Failure-streak inactivity | At 150s under validation-FPR calibration, mean forest recall was 0.730 without expiration and 0.184 at a 60s timeout | [Timeouts](../reports/streak_timeout/summary.csv) |
| Generator shift | Mean forest recall changed from 0.964 to 0.678 and FPR from 0.019 to 0.156 on a separately implemented generator | [Shift](../reports/shift/summary.csv) |
| Investigation capacity | On the already inspected 8M-12M range, 50 cases/day covered 3/25 known records for global context and 4/25 for personal context | [Public cases](../reports/rba_triage/summary.csv) |

The case review is retrospective. It uses frozen matched-budget cutoffs, the same user/source key, fixed 600-second windows from the first alert, and first-come admission. An endpoint alert opens a new case; windows do not extend. Overflow is permanent, admitted cases are monitored without extra cost across midnight, and no future score or label determines admission. Coverage requires the positive's own alert in an admitted case. The 10/50/100/unlimited capacity sensitivity is retained in the CSV; unlimited context coverage is 18/25. These work units do not estimate analyst hours or establish that 4/25 is operationally superior to 3/25.

## Interpretation

The external source is synthesized to preserve RBA-related categorical relationships. Unlabelled records are not verified negatives, and labelled rows may share an incident. Public results therefore report known-positive discovery and unlabelled volume, not precision, F1, true FPR, or production recall. Cold start, history contamination, and unresolved identity semantics remain limitations. Earlier inspected ranges become development evidence for subsequent designs.

The detailed historical reports, protocols, source archives, and audit logs are retained locally. The public tables summarize the recorded runs; new runs follow [Reproducibility](REPRODUCIBILITY.md).
