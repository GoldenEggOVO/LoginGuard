import pandas as pd
import pytest

from src.generate_data import generate
from src.features import build_features, FEATURE_COLUMNS
from src.rules import detect_rules
from src.evaluate import calculate_metrics
from src.model import chronological_split, fit_and_score


def events(rows):
    return pd.DataFrame(rows, columns=['event_id', 'timestamp', 'user_id', 'ip_address', 'device_id', 'status'])


def test_generator_reproducible_separate_labels_and_scenarios():
    a, labels = generate(seed=42)
    b, labels_b = generate(seed=42)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(labels, labels_b)
    assert 2000 <= len(a) <= 3000
    assert 20 <= a.user_id.nunique() <= 30
    assert set(a.columns) == {'event_id', 'timestamp', 'user_id', 'ip_address', 'device_id', 'status'}
    assert set(labels.attack_type) >= {'normal', 'brute_force', 'password_spray', 'suspicious_success'}
    assert not set(labels.columns[1:]) & set(a.columns)
    assert a.event_id.is_unique and labels.event_id.is_unique


def test_window_boundaries_and_device_history_are_causal():
    data = events([
        (1, '2024-01-01 00:00:00', 'u', 'ip', 'd', 'failure'),
        (2, '2024-01-01 00:01:00', 'u', 'ip', 'd', 'failure'),
        (3, '2024-01-01 00:01:01', 'v', 'ip', 'e', 'success'),
    ])
    f = build_features(data)
    assert f.account_failures_60s.tolist() == [1, 2, 0]
    assert f.ip_failures_60s.tolist() == [1, 2, 1]
    assert f.ip_distinct_users_10m.tolist() == [1, 1, 2]
    assert f.new_device.tolist() == [1, 0, 1]
    prefix = build_features(data.iloc[:2])
    pd.testing.assert_frame_equal(f.iloc[:2].reset_index(drop=True), prefix.reset_index(drop=True))
    assert 'is_anomaly' not in FEATURE_COLUMNS and 'attack_type' not in FEATURE_COLUMNS


def test_rules_independently_trigger_and_record_names():
    f = pd.DataFrame({'account_failures_60s':[5,0,0,0], 'ip_distinct_users_10m':[1,5,1,1],
                      'prior_consecutive_failures':[0,0,3,0], 'success':[0,0,1,1]})
    out = detect_rules(f)
    assert out.tolist() == ['repeated_failures', 'password_spraying', 'success_after_failures', '']


def test_chronological_split_no_overlap():
    times = pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-11', '2024-01-14'])
    train, val, test = chronological_split(times)
    assert train.tolist() == [True, False, False, False]
    assert val.tolist() == [False, True, False, False]
    assert test.tolist() == [False, False, True, True]


def test_metrics_known_confusion_matrix_and_zero_division():
    m = calculate_metrics([1,1,0,0], [1,0,1,0])
    assert (m['TP'],m['FP'],m['TN'],m['FN']) == (1,1,1,1)
    assert m['precision'] == m['recall'] == m['f1'] == m['fpr'] == m['fnr'] == .5
    assert calculate_metrics([0,0], [0,0])['precision'] == 0


def test_extra_label_columns_cannot_change_features_or_model_scores():
    data, labels = generate(seed=42)
    f = build_features(data)
    with_labels = data.merge(labels, on='event_id')
    pd.testing.assert_frame_equal(f, build_features(with_labels))
    train, val, _ = chronological_split(data.timestamp)
    scores, cutoff, _, _ = fit_and_score(f, train, val, labels.loc[val, 'is_anomaly'])
    contaminated = f.assign(is_anomaly=labels.is_anomaly, attack_type=labels.attack_type)
    scores2, cutoff2, _, _ = fit_and_score(contaminated, train, val, labels.loc[val, 'is_anomaly'])
    assert (scores == scores2).all()
    assert cutoff == cutoff2


def test_unsorted_events_are_rejected():
    data = events([(1, '2024-01-02', 'u', 'ip', 'd', 'success'),
                   (2, '2024-01-01', 'u', 'ip', 'd', 'failure')])
    with pytest.raises(ValueError, match='sorted'):
        build_features(data)


def test_ten_minute_boundary_and_tied_timestamps_use_input_order():
    data = events([(1, '2024-01-01 00:00:00', 'u', 'ip', 'd', 'success'),
                   (2, '2024-01-01 00:10:00', 'v', 'ip', 'e', 'failure'),
                   (3, '2024-01-01 00:10:00', 'u', 'ip', 'd', 'failure'),
                   (4, '2024-01-01 00:10:01', 'w', 'ip', 'f', 'failure')])
    f = build_features(data)
    assert f.ip_distinct_users_10m.tolist() == [1, 2, 2, 3]
    assert f.ip_failures_60s.tolist() == [0, 1, 2, 3]


def test_handwritten_sample_fields_and_valid_labels():
    sample = pd.read_csv('data/login_events_sample.csv')
    assert len(sample) == 10
    assert sample.notna().all().all()
    assert set(sample.status) == {'success', 'failure'}
    assert ((sample.attack_type == 'normal') == (sample.is_anomaly == 0)).all()
