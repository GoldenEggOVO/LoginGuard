import pandas as pd
import pytest

from src.generate_data import generate
from src.features import build_features
from src.model import chronological_split
from src.robustness import SEEDS, run_experiments


@pytest.mark.parametrize('seed', SEEDS)
def test_stress_preserves_training_validation_and_attack_counts(seed):
    original, labels = generate(seed)
    shifted, shifted_labels = generate(seed, scenario='familiar_slow')
    tr, val, _ = chronological_split(original.timestamp)
    tr2, val2, _ = chronological_split(shifted.timestamp)
    pd.testing.assert_frame_equal(original.loc[tr | val].reset_index(drop=True),
                                  shifted.loc[tr2 | val2].reset_index(drop=True))
    pd.testing.assert_frame_equal(labels.loc[tr | val].reset_index(drop=True),
                                  shifted_labels.loc[tr2 | val2].reset_index(drop=True))
    pd.testing.assert_series_equal(labels.attack_type.value_counts().sort_index(),
                                   shifted_labels.attack_type.value_counts().sort_index())
    test_attack = shifted.merge(shifted_labels, on='event_id')
    test_attack = test_attack[(test_attack.timestamp >= pd.Timestamp('2024-01-11')) & (test_attack.is_anomaly == 1)]
    assert all(row.device_id == f'device_{int(row.user_id.split("_")[1])}_0' for row in test_attack.itertuples())
    assert set(test_attack.ip_address) == {'198.51.100.200'}
    _, _, shifted_test = chronological_split(shifted.timestamp)
    assert build_features(shifted).loc[shifted_test & (shifted_labels.is_anomaly == 1), 'new_device'].sum() == 0
    with pytest.raises(ValueError):
        generate(42, scenario='unknown')


def test_experiment_is_paired_reproducible_and_validation_only(tmp_path):
    first = run_experiments(seeds=(42,), root=tmp_path)
    second = run_experiments(seeds=(42,), root=tmp_path)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 6
    assert set(first.scenario) == {'baseline', 'familiar_slow'}
    assert set(first.method) == {'Rules', 'Isolation Forest', 'Isolation Forest without new_device'}
    for method in ['Isolation Forest', 'Isolation Forest without new_device']:
        rows = first[first.method == method]
        assert rows.threshold.nunique() == 1
    assert (tmp_path / 'reports/robustness/per_seed.csv').exists()
    assert (tmp_path / 'reports/robustness/summary.csv').exists()
    by_attack = pd.read_csv(tmp_path / 'reports/robustness/per_attack.csv')
    assert len(by_attack) == 18
    assert set(by_attack.attack_type) == {'brute_force', 'password_spray', 'suspicious_success'}
    for row in first.itertuples():
        matching = by_attack[(by_attack.seed == row.seed) & (by_attack.scenario == row.scenario) & (by_attack.method == row.method)]
        assert matching.detected.sum() == row.TP
        assert matching.attack_events.sum() == row.test_attacks
