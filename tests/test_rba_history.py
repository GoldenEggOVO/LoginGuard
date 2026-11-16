import pandas as pd
from zipfile import ZipFile

from src.rba_history import history_features, run


def test_history_scores_current_event_before_updating_and_keeps_unknowns_distinct():
    events = pd.DataFrame({'user_id':['u','u','u','u','u'],
                           'ip_address':['ip1','ip1','ip2','ip2','ip2'],
                           'success':[1,0,0,1,0], 'device_type':['mobile','mobile','unknown','desktop','desktop']})
    h = history_features(events)
    assert h.has_prior_success.tolist() == [0,1,1,1,1]
    assert h.ip_seen_on_prior_success.tolist() == [0,1,0,0,1]
    assert h.has_prior_known_type.tolist() == [0,1,1,1,1]
    assert h.current_type_available.tolist() == [1,1,0,1,1]
    assert h.type_seen_on_prior_success.tolist() == [0,1,0,0,1]
    assert h.prior_success_count.tolist() == [0,1,1,1,2]
    pd.testing.assert_frame_equal(h.iloc[:3].reset_index(drop=True), history_features(events.iloc[:3]))


def test_holdout_stays_separate_and_labels_cannot_change_model(tmp_path):
    n = 500
    data = pd.DataFrame({'Login Timestamp': pd.date_range('2020-01-01', periods=n, freq='10s').astype(str),
                         'User ID': [str(i % 7) for i in range(n)],
                         'IP Address': [f'192.0.2.{1+i%8}' for i in range(n)],
                         'Device Type': ['mobile' if i % 2 else 'desktop' for i in range(n)],
                         'Login Successful': ['False' if i % 8 == 0 else 'True' for i in range(n)],
                         'Is Account Takeover': ['True' if i in (350, 450) else 'False' for i in range(n)]})
    archive = tmp_path / 'data.zip'
    with ZipFile(archive, 'w') as z:
        z.writestr('rba-dataset.csv', data.to_csv(index=False))
    first, audit1 = run(archive, max_rows=n, train_rows=300, development_rows=400,
                        retrospective_rows=(351,), root=tmp_path / 'one', expected_md5=None)
    assert len(first) == 2 and audit1['holdout_rows'] == 100
    assert (first.known_positives == 1).all()
    assert (first.validation_alert_rate <= .02).all()
    assert not {'precision','f1','fpr'} & set(first.columns)
    assert (tmp_path / 'one/reports/rba_history/missed_case_history.csv').exists()
    positives = pd.read_csv(tmp_path / 'one/reports/rba_history/holdout_positive_details.csv')
    assert len(positives) == 1 and positives.source_row.tolist() == [451]
    assert not {'user_id', 'ip_address', 'device_type'} & set(positives.columns)
    assert positives.baseline_alert.sum() == first.loc[first.model == 'baseline', 'known_positives_alerted'].iloc[0]
    import json
    identity = json.loads((tmp_path / 'one/reports/rba_history/identity_audit.json').read_text())
    assert {'validation_successes', 'holdout_successes', 'validation_unique_ips', 'holdout_unique_ips'} <= set(identity)
    segments = pd.read_csv(tmp_path / 'one/reports/rba_history/holdout_segments.csv')
    assert set(segments.segment) == {'no_prior_success', 'has_prior_success'}
    for row in first.itertuples():
        part = segments[segments.model == row.model]
        assert part.test_events.sum() == row.test_events
        assert part.alerts.sum() == row.alerts
        assert part.known_positives.sum() == row.known_positives
    data['Is Account Takeover'] = 'False'
    with ZipFile(archive, 'w') as z:
        z.writestr('rba-dataset.csv', data.to_csv(index=False))
    second, audit2 = run(archive, max_rows=n, train_rows=300, development_rows=400,
                         retrospective_rows=(351,), root=tmp_path / 'two', expected_md5=None)
    assert first.threshold.tolist() == second.threshold.tolist()
    assert first.alerts.tolist() == second.alerts.tolist()
    assert second.known_positive_recall.isna().all()
