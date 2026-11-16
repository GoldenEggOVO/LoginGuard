# Reproducibility

## Environment

The recorded runs use Windows/Python 3.14. Install `requirements-lock-py314.txt` to match package versions. Other Python 3.11+ environments can use `requirements.txt`; identical numerical results are not guaranteed. No GPU or hosted model API is required.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock-py314.txt
.venv/Scripts/python -B -m pytest -q -p no:cacheprovider
```

On POSIX use `python3 -m venv .venv`, `.venv/bin/python`, and `requirements.txt`. Run commands from the repository root with that environment's Python. `.gitattributes` preserves file bytes instead of applying Git line-ending conversion.

## Main experiments

```text
python -B -m src.pipeline
python -B -m src.fair_calibration
python -B -m src.rba_fair --archive data/public/rba/rba-dataset.zip
```

The baseline needs no external data. The two follow-ups use the [recorded protocol](fair_calibration_protocol.md). Public replay requires the separately downloaded archive and verifies its checksum; see [Data](DATA.md).

The new-seed comparison freezes source-seed calibration before generating evaluation streams. Public same-information evaluation fits only the original training successes, freezes calibration at row 2M, and evaluates only 12M+1-16M after carrying earlier history forward. These seeds and ranges are now inspected; rerunning them is reproduction, not new validation.

## Other entry points

| Study | Module |
| --- | --- |
| Paired factors | `src.mimicry` |
| Failure-history timeouts | `src.streak_timeout` |
| Synthetic case capacity | `src.triage_study` |
| Generator shift | `src.shift_study` |
| Initial public adaptation | `src.rba_validation` |
| Prior-success indicators | `src.rba_history` |
| Context selection/evaluation | `src.rba_context --phase development`, then `--phase holdout` |
| Matched calibration budget | `src.rba_budget` |
| Retrospective public cases | `src.rba_triage` |

The context selection must exist before dependent budget/case runs. The budget runner accepts the fixed integer parent boundaries 1.2M/2M/4M/8M and rejects unsupported protocols before replay. Public case admission follows the policy in [Experiments](EXPERIMENTS.md).

## Outputs and checks

Curated CSV results and required parameter files are in `reports/`; [Experiments](EXPERIMENTS.md) links the main tables. Full generated runs, fitted models, source snapshots, and historical audit material are kept locally. They are excluded from new public commits.

`src.pipeline` and early study runners overwrite their own output tables. Preserve published tables before rerunning. The context, budget, public-case, and latest follow-up runners write isolated `reports/<study>/runs/<id>/` outputs. A completion manifest is written before the latest-run pointer advances; failed runs cannot replace a completed pointer. Those local run directories include selections, model/source/protocol hashes and detailed metadata.

Tests cover temporal boundaries, label isolation, validation-only calibration, exact tied-score budgets, streaming equivalence, case admission, and failed-run preservation. GitHub Actions runs the bounded suite on Linux/Python 3.11 and 3.14 and Windows/Python 3.14, without downloading the public archive. Passing tests establish implementation checks, not detection effectiveness.

The recorded 16M public replay took 450.7 seconds and a 3.43 GiB process peak working set, using 100,000-row chunks and one forest worker. History dictionaries grow with observed categories; chunk size does not cap total memory. These measurements describe one machine and run.
