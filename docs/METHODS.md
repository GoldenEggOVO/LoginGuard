# Methods

## Task and features

LoginGuard analyzes completed authentication logs. The current login outcome is available. Features use the current record and earlier history; they do not predict authentication success before it happens. Labels are separate from feature computation.

| Synthetic feature | Definition |
| --- | --- |
| `hour` | Hour of the current timestamp |
| `success` | Current observed login outcome |
| `account_failures_60s` | Account failures in [t-60s, t], including the current failure |
| `ip_failures_60s` | Source failures in [t-60s, t], across accounts |
| `ip_distinct_users_10m` | Distinct accounts at the source in [t-600s, t], including successes |
| `new_device` | Device not previously observed for this account |
| `prior_consecutive_failures` | Account failure streak before the current result |

A success resets the streak after its prior value is read. An optional inactivity timeout resets it when time since the latest failure exceeds the limit; equality is retained. This is a maximum gap, not a rolling count. Synthetic device history includes failures, so previously seen does not imply trustworthy. Histories continue across splits and chunks. Stable input order resolves tied timestamps.

The public adaptation uses the five status/count features. It omits hour and unique-device novelty because the source has synthesized times and coarse device types.

## Rules and Isolation Forest

Fixed rules flag five account failures/60s, five distinct accounts/source/10min, or a success after three failures. These are heuristic reasons, not confirmed attack attribution. The calibrated control searches 245 threshold triples on validation labels and retains (2, 3, 5); ties follow the recorded enumeration.

Isolation Forest uses scikit-learn with 200 trees, random state 42, `contamination='auto'`, one worker, and automatic subsampling of at most 256 rows per tree. Larger `-score_samples` indicates greater anomaly. The synthetic cutoff maximizes validation F1 over deduplicated quantiles from 0.80 to 0.995. Fitting is unsupervised; choosing that cutoff uses labels. An anomaly score is not an attack probability.

## Context scoring

The six context fields are IP, ASN, country, OS, browser, and device type. For one field, G is the global successful count, K the observed category count, g(v) the current value count, n the user's successful count, and c(u,v) the user's value count:

$$p_g(v)=\frac{g(v)+1}{G+K+1},\qquad p_u(v)=\frac{c(u,v)+\alpha p_g(v)}{n+\alpha}.$$

The global support includes K observed values and one unseen-value bucket. Personal smoothing falls back to the global probability with no user history. Alpha=10 was selected using earlier development labels and is fixed in the latest comparison.

Mean surprisal averages -log(p) over available fields. Missing/unknown values are skipped, each field has its own denominator, and successful records are scored before frequency updates. Failures do not update successful-context counts. The all-unavailable score is zero by implementation definition; valid public records require an IP, so that case was not an observed evaluation subgroup.

For same-information controls, each forest receives six per-field surprisals and six availability indicators; the corresponding mean uses that same vector. The three new forests fit the same successful training records. Related fields may duplicate evidence, unidentified successful attacks can contaminate history, and the score is not a joint likelihood or a full Freeman-model reproduction.

## Calibration and evaluation

Synthetic metrics use generative attack labels. Public calibration uses all eligible successful records, not verified negatives. The latest comparison fixes budgets of 0.5%, 1%, and 2%, with 2% primary. A descending score order and deterministic SplitMix64 row key select exactly floor(b*n) calibration records. The saved boundary score/key applies to later records without ranking future batches. Equal calibration counts do not guarantee equal future alert volume or true FPR.

Public reports measure known-positive discovery and unlabelled alert volume. They cannot recover conventional precision, F1, or true FPR. Known rows may be related incidents. The full process uses development labels for model selection and is not completely unsupervised.

## Investigation cases

Alerts for the same account/source join a fixed 600-second window opened by its first alert; the endpoint opens a new case and later alerts do not extend it. New cases are admitted first-come under daily capacity, with permanent overflow. Admission uses no labels or future scores. Admitted cases are assumed monitored for later members at no extra cost, including across midnight. This is a simplified work-unit model, not analyst hours. A positive is covered only if its own alert belongs to an admitted case.

Source-window updates use incremental counts in public replay. Context storage grows with observed user/category pairs; chunking limits raw-table residency, not total history memory.
