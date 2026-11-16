import numpy as np
import pandas as pd

def test_rule_calibration_uses_validation_labels_and_deterministic_ties():
    from src.fair_calibration import select_rule_thresholds
    features = pd.DataFrame({'account_failures_60s': [0, 2, 3, 0],
                             'ip_distinct_users_10m': [1, 1, 1, 3],
                             'success': [1, 0, 0, 1],
                             'prior_consecutive_failures': [0, 0, 1, 0]})
    selected, candidates = select_rule_thresholds(features, np.array([0, 1, 1, 0]))
    assert len(candidates) == 245
    assert selected == {'failure_limit': 2, 'spray_limit': 4, 'success_failure_limit': 1}
    assert candidates.f1.max() == 1
    changed, _ = select_rule_thresholds(features, np.array([0, 0, 0, 1]))
    assert changed != selected


def test_rule_search_does_not_read_unselected_rows():
    from src.fair_calibration import select_rule_thresholds
    features = pd.DataFrame({'account_failures_60s': [0, 2, 0],
                             'ip_distinct_users_10m': [1, 1, 1],
                             'success': [1, 0, 1], 'prior_consecutive_failures': [0, 0, 0]})
    selected, table = select_rule_thresholds(features.iloc[:2], [0, 1])
    features.iloc[2] = [999, 999, 1, 999]
    repeated, repeated_table = select_rule_thresholds(features.iloc[:2], [0, 1])
    assert selected == repeated
    pd.testing.assert_frame_equal(table, repeated_table)


def test_synthetic_selection_precedes_new_seed_generation(tmp_path, monkeypatch):
    from src import fair_calibration
    from src.experiment_runs import resolve_run
    assert 307 not in fair_calibration.SEEDS
    original = fair_calibration.generate
    seen = []
    def generate(seed, **kwargs):
        if seed != 42:
            paths = list((tmp_path / 'reports/fair_calibration/runs').glob('*/selection.json'))
            assert len(paths) == 1
            seen.append(seed)
        return original(seed, **kwargs)
    monkeypatch.setattr(fair_calibration, 'generate', generate)
    monkeypatch.setattr(fair_calibration, 'SEEDS', (307,))
    results = fair_calibration.run(root=tmp_path)
    assert seen == [307, 307]
    assert len(results) == 6
    assert len(results.groupby('profile').attack_events.nunique()) == 2
    assert (resolve_run(tmp_path / 'reports/fair_calibration') / 'summary.csv').is_file()
