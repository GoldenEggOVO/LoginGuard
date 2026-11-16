import numpy as np
import pandas as pd
import pytest
from zipfile import ZipFile

from src.rba_validation import adapt_rows, chronological_features, alert_budget_cutoff, run
from src.features import build_features
from src.generate_data import generate


def test_adapter_preserves_unknown_labels_and_rejects_missing_required_fields():
    raw = pd.DataFrame({
        'Login Timestamp': ['2020-02-03 12:00:00', '2020-02-03 12:01:00', 'bad', '2020-02-03 12:02:00'],
        'User ID': ['100', '200', '300', ''],
        'IP Address': ['192.0.2.1', '198.51.100.1', '192.0.2.2', '203.0.113.1'],
        'Login Successful': ['False', 'True', 'True', 'False'],
        'Is Account Takeover': ['True', 'False', '', 'True'],
    })
    events, labels, audit = adapt_rows(raw)
    assert len(events) == 2 and audit['excluded_rows'] == 2
    assert labels.label_state.tolist() == ['known_positive', 'unlabelled_explicit_false']
    assert events.success.tolist() == [0, 1]
    assert not {'device_id', 'new_device', 'Is Account Takeover'} & set(events.columns)
    assert events.source_row.tolist() == [1, 2]


def test_features_include_boundaries_without_future_or_device():
    data = pd.DataFrame({'timestamp': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:01:00',
                                                      '2020-01-01 00:01:00', '2020-01-01 00:01:01']),
                         'user_id':['u','v','u','u'], 'ip_address':['ip','ip','ip','ip'], 'success':[0,0,1,0]})
    f = chronological_features(data)
    assert f.account_failures_60s.tolist() == [1,1,1,1]
    assert f.ip_failures_60s.tolist() == [1,2,2,2]
    assert f.ip_distinct_users_10m.tolist() == [1,2,2,2]
    assert f.prior_consecutive_failures.tolist() == [0,0,1,0]
    pd.testing.assert_frame_equal(f.iloc[:3].reset_index(drop=True), chronological_features(data.iloc[:3]))
    assert 'new_device' not in f
    with pytest.raises(ValueError, match='chronological'):
        chronological_features(data.iloc[::-1])


def test_validation_alert_budget_handles_ties_without_using_labels():
    scores = np.array([.1] * 97 + [.9] * 3)
    assert (scores >= alert_budget_cutoff(scores)).mean() <= .02
    scores = np.arange(100, dtype=float)
    assert (scores >= alert_budget_cutoff(scores)).mean() == .02


def test_compatible_features_match_original_on_synthetic_events():
    events, _ = generate(42)
    adapted = events.assign(success=events.status.eq('success').astype(int))
    reference = build_features(events)
    result = chronological_features(adapted)
    pd.testing.assert_frame_equal(result, reference[result.columns], check_dtype=False)


def test_end_to_end_keeps_labels_out_of_model_and_reports_unknowns(tmp_path):
    n = 300
    raw = pd.DataFrame({
        'Login Timestamp': pd.date_range('2020-01-01', periods=n, freq='10s').astype(str),
        'User ID': [str(i % 7) for i in range(n)],
        'IP Address': [f'192.0.2.{1 + i % 8}' for i in range(n)],
        'Login Successful': ['False' if i % 8 == 0 else 'True' for i in range(n)],
        'Is Account Takeover': ['True' if i == 290 else 'False' for i in range(n)],
    })
    archive = tmp_path / 'sample.zip'
    with ZipFile(archive, 'w') as z:
        z.writestr('rba-dataset.csv', raw.to_csv(index=False))
    with pytest.raises(ValueError, match='MD5'):
        run(archive, max_rows=n, root=tmp_path / 'bad', expected_md5='0' * 32)
    first, audit1 = run(archive, max_rows=n, root=tmp_path / 'one', expected_md5=None)
    assert audit1['known_positives_by_split']['test'] == 1
    assert audit1['validation_alert_rate_all_events'] <= .02
    assert not {'precision', 'f1', 'fpr', 'false_positives'} & set(first.columns)
    assert (first.known_positives == 1).all()
    breakdown = pd.read_csv(tmp_path / 'one/reports/rba/rule_breakdown.csv')
    assert set(breakdown.rule) == {'repeated_failures', 'multiple_accounts_per_ip', 'success_after_failures'}
    positive_rows = pd.read_csv(tmp_path / 'one/reports/rba/known_positive_rows.csv')
    assert {'account_failures_60s', 'ip_distinct_users_10m', 'prior_consecutive_failures'} <= set(positive_rows.columns)
    sensitivity = pd.read_csv(tmp_path / 'one/reports/rba/metrics_without_train_top_id.csv')
    assert (sensitivity.test_events < first.test_events).all()
    assert (tmp_path / 'one/reports/rba/identity_audit.json').exists()
    raw['Is Account Takeover'] = 'False'
    with ZipFile(archive, 'w') as z:
        z.writestr('rba-dataset.csv', raw.to_csv(index=False))
    second, audit2 = run(archive, max_rows=n, root=tmp_path / 'two', expected_md5=None)
    assert audit1['threshold'] == audit2['threshold']
    assert first.alerts.tolist() == second.alerts.tolist()
    assert second.known_positive_recall.isna().all()
