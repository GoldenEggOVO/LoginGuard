# LoginGuard

LoginGuard studies authentication anomaly detection when attacks slow down, reuse familiar context, or compete for a limited investigation budget. It compares rules, Isolation Forest, and probabilistic context scores using chronological login streams.

The project focuses on controlled comparisons and error analysis. It uses existing algorithms, synthetic streams, and a published synthesized authentication dataset. Its results do not establish production performance.

## Quick start

Recorded environment: Windows, Python 3.14.

```powershell
git clone https://github.com/GoldenEggOVO/LoginGuard.git
cd LoginGuard
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock-py314.txt
.venv/Scripts/python -B -m pytest -q -p no:cacheprovider
.venv/Scripts/python -m src.pipeline
```

The baseline generates 2,331 login events and uses a seven-day training, three-day validation, four-day test split. It writes predictions, metrics, and plots to `reports/`; rerunning it replaces its baseline output files. Tests need no external dataset. For POSIX and other Python 3.11+ environments, use `requirements.txt` and the commands in [Reproducibility](docs/REPRODUCIBILITY.md).

## Results

Giving rules the same validation-label access narrows the original gap. On eight new generator seeds, mean event F1 is:

| Test profile | Fixed rules | Calibrated rules | Isolation Forest |
| --- | ---: | ---: | ---: |
| Unfamiliar device, 20s, attack-specific IP | 0.144 | 0.822 | 0.939 |
| Primary device, 150s, shared IP | 0.166 | 0.286 | 0.724 |

The latest public-data evaluation uses records 12,000,001-16,000,000. At the prespecified 2% calibration budget:

| Method | Known-positive records detected | Unlabelled alerts |
| --- | ---: | ---: |
| Success-fitted count forest | 0/30 | 41,746 |
| Global context mean | 29/30 | 59,632 |
| Personal context mean | 29/30 | 42,630 |
| Forest on the same personal-context representation | 29/30 | 47,309 |

Context representation matters in this source; the more complex scorer does not improve the primary personal-context result. Equal detection counts do not imply identical alert sets. Labels are incomplete, so unlabelled alerts are not confirmed false positives. More findings, controls, and result tables are in [Experiments](docs/EXPERIMENTS.md).

## Documentation

[Methods](docs/METHODS.md) · [Data](docs/DATA.md) · [Experiments](docs/EXPERIMENTS.md) · [Reproducibility](docs/REPRODUCIBILITY.md)

```text
src/        Features, detectors, experiment runners, and case admission
tests/      Temporal boundaries, calibration, label isolation, and run integrity
scripts/    Data inspection, diagnostics, plotting, and label audits
docs/       Research methods, data definitions, results, and running instructions
reports/    Result tables and required experiment parameters
data/       Small synthetic examples; external archives are excluded
```

Full generated runs, model snapshots, and historical research archives stay local. Newer runners save isolated outputs; the public repository keeps the result tables needed to inspect the comparisons.

## References and contributions

The [Login Data Set for Risk-Based Authentication](https://zenodo.org/records/6782156) is CC BY 4.0; cite [Wiefling et al. (2022)](https://doi.org/10.1145/3546069) when using it. Context scoring draws on probabilistic RBA work, including [Unsel et al. (2023)](https://doi.org/10.1145/3577923.3583634), with the simplifications described in Methods. It is not a full Freeman-model reproduction. Isolation Forest uses scikit-learn.

The project owner directed the research questions and requirements, including behavioral mimicry and failure-history limits. Codex assisted with implementation, experiment execution, testing, review, error analysis, and documentation.
