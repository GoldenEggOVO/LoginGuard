import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from src.rba_validation import chronological_features
from src.experiment_runs import resolve_run


def test_window_state_across_chunks_matches_whole_stream():
    events = pd.DataFrame({'timestamp': pd.to_datetime(['2020-01-01 00:00:00','2020-01-01 00:01:00',
                                                       '2020-01-01 00:01:00','2020-01-01 00:11:01']),
                           'user_id':['u','v','u','u'], 'ip_address':['ip']*4, 'success':[0,0,1,0]})
    state = {}
    chunks = [chronological_features(events.iloc[:2], state), chronological_features(events.iloc[2:], state)]
    pd.testing.assert_frame_equal(pd.concat(chunks, ignore_index=True), chronological_features(events))
    with pytest.raises(ValueError, match='chronological'):
        chronological_features(events.iloc[:1], state)


def test_context_scores_prior_only_and_backoff_handles_unknown_values():
    from src.rba_context import ContextRisk, CONTEXT_COLUMNS
    model = ContextRisk()
    normal = ['192.0.2.1','1','US','ExampleOS','ExampleBrowser','desktop']
    first, prior = model.score_update('u', normal, True)
    assert prior == 0 and all(np.isfinite(list(first.values())) > 0)
    repeat, prior = model.score_update('u', normal, False)
    assert prior == 1
    assert repeat['global_surprisal'] == pytest.approx(-np.log(2/3))
    assert repeat['surprisal_a1'] == pytest.approx(-np.log(5/6))
    novel, _ = model.score_update('u', ['x','2','JP','OtherOS','OtherBrowser','mobile'], False)
    assert novel['surprisal_a1'] > repeat['surprisal_a1']
    model.score_update('u', ['192.0.2.2','2','JP','OtherOS','OtherBrowser','mobile'], False)
    again, prior = model.score_update('u', normal, False)
    assert again == repeat and prior == 1
    missing, _ = model.score_update('u', ['unknown'] * len(CONTEXT_COLUMNS), False)
    assert all(v == 0 for v in missing.values())
    other, prior = model.score_update('v', normal, False)
    assert prior == 0 and other['surprisal_a1'] == other['global_surprisal']
    assert other['ratio_a1'] == other['global_surprisal']


def test_development_selection_is_not_changed_by_holdout_labels_or_context(tmp_path):
    from src.rba_context import run
    n = 800
    raw = pd.DataFrame({'Login Timestamp':pd.date_range('2020-01-01', periods=n, freq='10s').astype(str),
                        'User ID':[str(i % 7) for i in range(n)],
                        'IP Address':[f'192.0.2.{1+i%8}' for i in range(n)],
                        'Login Successful':['False' if i%8 == 0 else 'True' for i in range(n)],
                        'Is Account Takeover':['True' if i in (450,650) else 'False' for i in range(n)],
                        'ASN':['1']*n,'Country':['US']*n,'OS Name and Version':['OS']*n,
                        'Browser Name and Version':['Browser']*n,'Device Type':['desktop']*n})
    archive = tmp_path / 'sample.zip'
    def save():
        with ZipFile(archive,'w') as z:
            z.writestr('rba-dataset.csv',raw.to_csv(index=False))
    save()
    settings = dict(archive_path=archive, train_rows=200, calibration_rows=400,
                    development_rows=600, max_rows=n, chunksize=100, expected_md5=None)
    one = tmp_path / 'one'
    run(root=one, **settings)
    one_output = resolve_run(one/'reports/rba_context')
    selection = json.loads((one_output/'selection.json').read_text())
    metrics = pd.read_csv(one_output/'holdout_metrics.csv')
    assert set(metrics.model) == {'forest_all','forest_success','global_surprisal',selection['selected']}
    assert (metrics.known_positives == 1).all()
    assert not {'precision','f1','fpr'} & set(metrics)
    success_rows = metrics[metrics.model != 'forest_all']
    assert success_rows.failure_alerts.sum() == 0
    assert (success_rows.calibration_success_alert_rate <= .02).all()
    details = pd.read_csv(one_output/'holdout_positive_details.csv')
    assert not {'user_id','ip_address','Country','ASN'} & set(details)
    raw.loc[600:, 'Is Account Takeover'] = 'False'
    raw.loc[600:, 'Country'] = 'JP'
    save()
    two = tmp_path / 'two'
    run(root=two, **settings)
    second = json.loads((resolve_run(two/'reports/rba_context')/'selection.json').read_text())
    assert selection['selected'] == second['selected']
    assert selection['thresholds'] == second['thresholds']
    assert selection['development_metrics'] == second['development_metrics']


def test_context_replay_is_chunk_independent_and_future_invariant():
    from src.rba_context import ContextRisk
    prefix = [('u',['a','1','US','OS','B','desktop'],True),
              ('v',['b','2','JP','OS','B','mobile'],True),
              ('u',['b','2','JP','OS','B','desktop'],False)]
    first = ContextRisk()
    rows = [first.score_update(*event) for event in prefix]
    second = ContextRisk()
    assert [second.score_update(*event) for event in prefix[:2]] == rows[:2]
    assert second.score_update(*prefix[2]) == rows[2]


def test_missing_context_does_not_inflate_personal_valid_count():
    from src.rba_context import ContextRisk
    model = ContextRisk()
    missing = ['192.0.2.1','1','US','','unknown','unknown']
    model.score_update('u',missing,True)
    assert model.successes['u'] == 1
    assert model.user_totals[3]['u'] == 0
    assert model.user_totals[0]['u'] == 1


def test_candidate_selection_only_uses_development_labels():
    from src.rba_context import develop, CANDIDATES
    n = 1000
    data = {name:np.linspace(0,1,n) for name in CANDIDATES}
    data.update(forest=np.linspace(0,1,n), source_row=np.arange(201,1201),
                success=np.ones(n,dtype=bool), known=np.zeros(n,dtype=bool))
    one = develop(data,200,600,1000)
    data['known'][:400] = True
    data['known'][800:] = True
    two = develop(data,200,600,1000)
    assert one == two


def test_replay_matches_with_arbitrary_chunk_size_and_rejects_time_tie_at_split(tmp_path):
    from src.rba_context import replay
    n = 80
    data = pd.DataFrame({'Login Timestamp':pd.date_range('2020-01-01',periods=n,freq='1s').astype(str),
                         'User ID':['1']*n,'IP Address':['192.0.2.1']*n,
                         'Login Successful':['True']*n,'Is Account Takeover':['False']*n,
                         'ASN':['1']*n,'Country':['US']*n,'OS Name and Version':['OS']*n,
                         'Browser Name and Version':['Browser']*n,'Device Type':['desktop']*n})
    archive = tmp_path/'chunks.zip'
    def save():
        with ZipFile(archive,'w') as z:
            z.writestr('rba-dataset.csv',data.to_csv(index=False))
    save()
    a,_ = replay(archive,80,20,40,60,10)
    b,_ = replay(archive,80,20,40,60,13)
    for key in a:
        np.testing.assert_array_equal(a[key],b[key])
    data.loc[20,'Login Timestamp'] = data.loc[19,'Login Timestamp']
    save()
    with pytest.raises(ValueError,match='equal timestamps'):
        replay(archive,80,20,40,60,13)


def test_invalid_boundary_chunk_does_not_bypass_time_tie_check(tmp_path):
    from src.rba_context import replay
    n = 80
    data = pd.DataFrame({'Login Timestamp':pd.date_range('2020-01-01',periods=n,freq='1s').astype(str),
                         'User ID':['1']*n,'IP Address':['192.0.2.1']*n,
                         'Login Successful':['True']*n,'Is Account Takeover':['False']*n,
                         'ASN':['1']*n,'Country':['US']*n,'OS Name and Version':['OS']*n,
                         'Browser Name and Version':['Browser']*n,'Device Type':['desktop']*n})
    data.loc[20:30,'Login Timestamp'] = data.loc[19,'Login Timestamp']
    data.loc[20:29,'IP Address'] = 'invalid'
    archive = tmp_path/'invalid_boundary.zip'
    with ZipFile(archive,'w') as z:
        z.writestr('rba-dataset.csv',data.to_csv(index=False))
    for chunksize in (10,80):
        with pytest.raises(ValueError,match='equal timestamps'):
            replay(archive,80,20,40,60,chunksize)


def test_additional_holdout_boundary_is_checked_without_changing_training(tmp_path):
    from src.rba_context import replay
    n = 120
    data = pd.DataFrame({'Login Timestamp':pd.date_range('2020-01-01',periods=n,freq='1s').astype(str),
                         'User ID':['1']*n,'IP Address':['192.0.2.1']*n,
                         'Login Successful':['True']*n,'Is Account Takeover':['False']*n,
                         'ASN':['1']*n,'Country':['US']*n,'OS Name and Version':['OS']*n,
                         'Browser Name and Version':['Browser']*n,'Device Type':['desktop']*n})
    archive = tmp_path/'extra_boundary.zip'
    def save():
        with ZipFile(archive,'w') as z:
            z.writestr('rba-dataset.csv',data.to_csv(index=False))
    save()
    first,_ = replay(archive,120,12,20,40,13)
    second,_ = replay(archive,120,12,20,40,13,extra_boundaries=(80,))
    for key in first:
        np.testing.assert_array_equal(first[key],second[key])
    data.loc[80,'Login Timestamp'] = data.loc[79,'Login Timestamp']
    save()
    with pytest.raises(ValueError,match='equal timestamps'):
        replay(archive,120,12,20,40,13,extra_boundaries=(80,))
