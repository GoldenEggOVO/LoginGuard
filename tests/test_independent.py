import pandas as pd

from src.independent_data import generate_independent
from src.model import chronological_split
from src.shift_study import run_study


def test_independent_generator_reproducible_labels_separate_and_temporal():
    events, labels, episodes = generate_independent(241)
    other_events, other_labels, other_episodes = generate_independent(241)
    pd.testing.assert_frame_equal(events, other_events)
    pd.testing.assert_frame_equal(labels, other_labels)
    pd.testing.assert_frame_equal(episodes, other_episodes)
    assert set(events.columns) == {'event_id', 'timestamp', 'user_id', 'ip_address', 'device_id', 'status'}
    assert events.event_id.equals(labels.event_id) and events.event_id.equals(episodes.event_id)
    assert events.timestamp.is_monotonic_increasing and events.event_id.is_unique
    assert 20 <= events.user_id.nunique() <= 30
    assert set(labels.attack_type) == {'normal', 'brute_force', 'password_spray', 'suspicious_success'}
    assert (labels.is_anomaly == episodes.episode_id.notna().astype(int)).all()
    train, validation, test = chronological_split(events.timestamp)
    assert labels.loc[train | validation, 'is_anomaly'].sum() == 0
    assert labels.loc[test, 'is_anomaly'].sum() > 0
    assert len(episodes.loc[test, 'episode_id'].dropna().unique()) == 12


def test_transfer_study_freezes_source_model_and_writes_metrics(tmp_path):
    results = run_study(seeds=(241,), budget=5, root=tmp_path)
    assert len(results) == 6
    assert set(results.dataset) == {'original', 'independent'}
    assert set(results.method) == {'Rules', 'Isolation Forest', 'Union'}
    assert results.threshold.nunique() == 1
    assert (results.cases == results.reviewed_cases + results.overflow_cases).all()
    assert (results.episodes == results.episodes_covered + results.episodes_missed).all()
    assert (tmp_path / 'reports/shift/per_seed.csv').exists()
    assert (tmp_path / 'reports/shift/summary.csv').exists()
    diagnostics = pd.read_csv(tmp_path / 'reports/shift/normal_diagnostics.csv')
    for row in diagnostics.itertuples():
        metric = results[(results.dataset == row.dataset) & (results.method == 'Isolation Forest')].iloc[0]
        assert row.normal_events == metric.TN + metric.FP
        assert row.alerted_normal_events == metric.FP
