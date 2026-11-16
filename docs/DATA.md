# Data

## Synthetic streams

The baseline generator produces 2,331 events for 25 users across 14 days at seed 42. Seven normal-only days fit the forest, three days calibrate, and four days evaluate. Attack stories cover brute force, password spraying, and success after failures. Early attempts keep their attack labels even when they are indistinguishable from ordinary failures.

`data/login_events.csv` and `data/ground_truth.csv` are generated baseline tables. `login_events_sample.csv` is an illustrative ten-row example, not the experiment input. The original generator resamples spacing in `base + j*spacing`; controlled profiles use fixed offsets. A separately implemented generator adds user rhythms, retry clusters, and different activity rates. Both remain project-designed simulations.

The latest comparison fits/calibrates on reference seed 42, then evaluates seeds 311, 313, 317, 331, 337, 347, 349, and 353. Each target's first ten days supply history only. Seed 307 is a development fixture, excluded by the recorded amendment in the [protocol](fair_calibration_protocol.md).

## External authentication source

[Login Data Set for Risk-Based Authentication](https://zenodo.org/records/6782156), Wiefling et al., CC BY 4.0. Cite [the dataset paper](https://doi.org/10.1145/3546069). Obtain `rba-dataset.zip` separately and place it at `data/public/rba/rba-dataset.zip`.

- Publisher MD5: `cc1b1078b3929650e6c08678caffcc57`
- Recorded SHA-256: `ca1d974e97aebfb30878a613f4ca5c793860a98ba2acb5185c9bc610d7432a33`

The archive is about 1.1 GB. Feature values are synthesized from real-service statistics to preserve relationships useful for RBA. Times and IPs are artificial; these are not raw production logs. Raw archives are excluded from GitHub.

| Source field | Treatment |
| --- | --- |
| User ID | Grouping pseudonym; the high-frequency identifier's meaning remains unknown |
| Login timestamp | Stream ordering; no real travel inference |
| IP address | Validated grouping/context value; never contacted |
| Login Successful | Current observed status, not an attack label |
| Device Type | Coarse category, not a unique physical device |
| Is Account Takeover | True is a known positive; remaining records stay unlabelled |
| Is Attack IP, RTT | Excluded from model inputs |

## Public study ranges

| Study | Development basis | Evaluation records |
| --- | --- | --- |
| Initial adaptation | First 1.2M fit; next 0.4M calibrate | 1,600,001-2,000,000: 4 known positives |
| Prior-success features | First 2M | 2,000,001-4,000,000: 9 known positives |
| Global/personal context | First 1.2M fit; next 0.8M calibrate; next 2M select | 4,000,001-8,000,000: 24 known positives |
| Matched budget | Earlier development only | 8,000,001-12,000,000: 25 known positives |
| Case admission | Frozen methods, already inspected outcomes | Same 8M-12M range, retrospectively |
| Same-information controls | Training successes in first 1.2M; calibration successes in 1.2M-2M | 12,000,001-16,000,000: 30 known positives |

The latest range has 4M records and 1,789,973 successes. All 30 known positives have successful status. One of the earlier 25 positives has failed status and remains in that study's denominator. No labels were rewritten or silently removed.

Raw ordinals define processing. Internal train/calibration/evaluation-start boundaries reject equal timestamps; the latest runner checks before adaptation. The 16M terminal cap does not certify a complete incident group. Earlier records continue to update causal history. Once inspected, a range is development evidence for subsequent designs.

Unlabelled does not mean normal. Labels may reflect incident-response ascertainment, related records, and unknown coverage. The unresolved pseudonym is retained without a special semantic interpretation. LANL and AIT were assessed for feasibility; they are not additional completed detection benchmarks.
