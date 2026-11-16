import numpy as np
import pandas as pd
import pytest

from src.features import build_features
from src.generate_data import generate
from src.model import chronological_split
from src.streak_timeout import SEEDS, run_study, validation_fpr_cutoff


def test_timeout_expires_after_not_at_boundary_and_success_resets():
    events = pd.DataFrame([
        (1, '2024-01-01 00:00:00', 'u', 'ip', 'd', 'failure'),
        (2, '2024-01-01 00:00:30', 'v', 'ip', 'e', 'failure'),
        (3, '2024-01-01 00:01:00', 'u', 'ip', 'd', 'failure'),
        (4, '2024-01-01 00:02:01', 'u', 'ip', 'd', 'failure'),
        (5, '2024-01-01 00:03:01', 'u', 'ip', 'd', 'success'),
        (6, '2024-01-01 00:03:01', 'u', 'ip', 'd', 'failure'),
        (7, '2024-01-01 00:04:01.001', 'u', 'ip', 'd', 'success'),
    ], columns=['event_id', 'timestamp', 'user_id', 'ip_address', 'device_id', 'status'])
    events['timestamp'] = pd.to_datetime(events['timestamp'], format='mixed')
    original = build_features(events)
    limited = build_features(events, failure_gap_seconds=60)
    assert original.prior_consecutive_failures.tolist() == [0, 0, 1, 2, 3, 0, 1]
    assert limited.prior_consecutive_failures.tolist() == [0, 0, 1, 0, 1, 0, 0]
    pd.testing.assert_frame_equal(original.drop(columns='prior_consecutive_failures'),
                                  limited.drop(columns='prior_consecutive_failures'))
    pd.testing.assert_frame_equal(limited.iloc[:4].reset_index(drop=True),
                                  build_features(events.iloc[:4], failure_gap_seconds=60))
    for value in (0, -1, float('nan'), float('inf'), True):
        with pytest.raises(ValueError):
            build_features(events, failure_gap_seconds=value)


def test_fpr_cutoff_enforces_budget_with_ties():
    labels = np.zeros(100, dtype=int)
    scores = np.array([.1] * 97 + [.9] * 3)
    threshold = validation_fpr_cutoff(scores, labels)
    assert np.mean(scores >= threshold) <= .02
    scores = np.arange(100, dtype=float)
    threshold = validation_fpr_cutoff(scores, labels)
    assert np.mean(scores >= threshold) == .02
    with pytest.raises(ValueError):
        validation_fpr_cutoff([.2, .3], [1, 1])


@pytest.mark.parametrize('seed', SEEDS)
def test_attack_gaps_are_paired_and_devices_known(seed):
    baseline, labels, episodes = generate(seed, test_profile=(20, True, False), include_episodes=True)
    train, validation, test = chronological_split(baseline.timestamp)
    assert build_features(baseline).loc[test & labels.is_anomaly.eq(1), 'new_device'].sum() == 0
    reference = baseline.merge(labels, on='event_id').merge(episodes, on='event_id')
    reference = reference[reference.episode_id.notna()].set_index(['episode_id', 'episode_step']).sort_index()
    for gap in (60, 150):
        candidate, truth, meta = generate(seed, test_profile=(gap, True, False), include_episodes=True)
        tr, val, test_candidate = chronological_split(candidate.timestamp)
        assert build_features(candidate).loc[test_candidate & truth.is_anomaly.eq(1), 'new_device'].sum() == 0
        pd.testing.assert_frame_equal(baseline.loc[train | validation].reset_index(drop=True),
                                      candidate.loc[tr | val].reset_index(drop=True))
        pd.testing.assert_frame_equal(labels.loc[train | validation].reset_index(drop=True),
                                      truth.loc[tr | val].reset_index(drop=True))
        paired = candidate.merge(truth, on='event_id').merge(meta, on='event_id')
        paired = paired[paired.episode_id.notna()].set_index(['episode_id', 'episode_step']).sort_index()
        pd.testing.assert_frame_equal(reference[['user_id', 'status', 'attack_type', 'ip_address', 'device_id']],
                                      paired[['user_id', 'status', 'attack_type', 'ip_address', 'device_id']])


def test_study_has_paired_fixed_thresholds_and_reproducible_outputs(tmp_path):
    first = run_study(seeds=(149,), root=tmp_path)
    second = run_study(seeds=(149,), root=tmp_path)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 27
    assert set(first.timeout) == {'none', '60s', '300s'}
    assert set(first.calibration) == {'fixed_rules', 'validation_f1', 'validation_fpr_2pct'}
    assert first.groupby(['timeout', 'calibration']).threshold.nunique(dropna=True).max() == 1
    assert (first.episodes == 16).all()
    assert (first.episodes == first.episodes_detected + first.episodes_undetected).all()
    assert (first.loc[first.calibration == 'validation_fpr_2pct', 'validation_fpr'] <= .02).all()
    assert (tmp_path / 'reports/streak_timeout/per_seed.csv').exists()
    assert (tmp_path / 'reports/streak_timeout/per_episode.csv').exists()
    per_attack = pd.read_csv(tmp_path / 'reports/streak_timeout/per_attack.csv')
    per_episode = pd.read_csv(tmp_path / 'reports/streak_timeout/per_episode.csv')
    for row in first.itertuples():
        select = lambda frame: ((frame.seed == row.seed) & (frame.timeout == row.timeout)
                                & (frame.gap_seconds == row.gap_seconds) & (frame.calibration == row.calibration))
        attacks, episodes = per_attack[select(per_attack)], per_episode[select(per_episode)]
        assert attacks.detected.sum() == row.TP
        assert attacks.attack_events.sum() == row.test_attacks
        assert episodes.detected.sum() == row.episodes_detected
        assert episodes.loc[episodes.detected == 0, 'latency_seconds'].isna().all()
