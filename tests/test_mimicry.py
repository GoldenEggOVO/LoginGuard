from itertools import product

import numpy as np
import pandas as pd
import pytest

from src.generate_data import generate
from src.features import build_features
from src.model import chronological_split, fit_and_score, fit_with_threshold
from src.mimicry import SEEDS, episode_stats, run_study


@pytest.mark.parametrize('seed', SEEDS)
def test_profile_changes_only_test_attack_dimensions_and_retains_pairing(seed):
    base, labels, episodes = generate(seed, test_profile=(20, False, False), include_episodes=True)
    tr, val, _ = chronological_split(base.timestamp)
    reference = base.merge(labels, on='event_id').merge(episodes, on='event_id')
    reference = reference[reference.episode_id.notna()].set_index(['episode_id', 'episode_step']).sort_index()
    for gap, device, source in product((20, 60, 150), (False, True), (False, True)):
        ev, lab, ep = generate(seed, test_profile=(gap, device, source), include_episodes=True)
        train, validation, test = chronological_split(ev.timestamp)
        pd.testing.assert_frame_equal(base.loc[tr | val].reset_index(drop=True), ev.loc[train | validation].reset_index(drop=True))
        pd.testing.assert_frame_equal(labels.loc[tr | val].reset_index(drop=True), lab.loc[train | validation].reset_index(drop=True))
        attack = ev.merge(lab, on='event_id').merge(ep, on='event_id')
        attack = attack[attack.episode_id.notna()].set_index(['episode_id', 'episode_step']).sort_index()
        pd.testing.assert_frame_equal(reference[['user_id', 'status', 'attack_type']], attack[['user_id', 'status', 'attack_type']])
        assert 'episode_id' not in ev and 'episode_step' not in ev
        test_attack = lab.is_anomaly.eq(1) & test
        if device:
            assert build_features(ev).loc[test_attack, 'new_device'].sum() == 0
        if source:
            assert set(ev.loc[test_attack, 'ip_address']) == {'198.51.100.200'}
        for _, batch in attack[attack.index.get_level_values('episode_id').str.startswith(('10_', '11_', '12_', '13_'))].groupby(level='episode_id'):
            assert batch.timestamp.diff().dropna().dt.total_seconds().eq(gap).all()
    with pytest.raises(ValueError):
        generate(103, test_profile=(0, True, False))


def test_episode_metrics_count_undetected_batches_and_latency():
    ev = pd.DataFrame({'event_id': [1, 2, 3, 4, 5], 'timestamp': pd.to_datetime([
        '2024-01-01 00:00:00', '2024-01-01 00:01:00', '2024-01-01 00:02:00',
        '2024-01-01 00:03:00', '2024-01-01 00:04:00'])})
    ep = pd.DataFrame({'event_id': [1, 2, 3, 4, 5], 'episode_id': ['a', 'a', 'b', 'b', None],
                       'episode_step': [0, 1, 0, 1, None]})
    summary, details = episode_stats(ev, ep, np.array([0, 1, 0, 0, 1]), np.ones(5, dtype=bool))
    assert summary['episodes'] == 2 and summary['episodes_detected'] == 1
    assert summary['episode_recall'] == .5 and summary['median_detected_seconds'] == 60
    assert details.set_index('episode_id').loc['b', 'detected'] == 0
    assert pd.isna(details.set_index('episode_id').loc['b', 'latency_seconds'])
    empty, _ = episode_stats(ev, ep, np.zeros(5), np.ones(5, dtype=bool))
    assert empty['episode_recall'] == 0 and np.isnan(empty['median_detected_seconds'])


def test_reusable_model_matches_previous_scores_and_threshold():
    events, labels = generate(103)
    f = build_features(events)
    tr, val, _ = chronological_split(events.timestamp)
    scores, threshold, _, _ = fit_and_score(f, tr, val, labels.loc[val, 'is_anomaly'])
    model, cutoff, _, _ = fit_with_threshold(f, tr, val, labels.loc[val, 'is_anomaly'])
    np.testing.assert_array_equal(scores, -model.score_samples(f))
    assert threshold == cutoff


def test_study_outputs_paired_conditions_and_episode_counts(tmp_path):
    results = run_study(seeds=(103,), root=tmp_path)
    assert len(results) == 24
    assert results.groupby('method').threshold.nunique(dropna=True).max() == 1
    assert (results.episodes == results.episodes_detected + results.episodes_undetected).all()
    assert (results.episodes == 16).all()
    assert (tmp_path / 'reports/mimicry/per_seed.csv').exists()
    assert (tmp_path / 'reports/mimicry/per_episode.csv').exists()
