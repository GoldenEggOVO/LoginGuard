import numpy as np
import pandas as pd
import pytest

from src.triage import group_alerts


def fixture_stream():
    times = pd.to_datetime(['2020-01-01 23:55','2020-01-02 00:01','2020-01-02 00:02',
                            '2020-01-02 00:05','2020-01-02 00:06','2020-01-02 00:07'])
    truth = pd.DataFrame({'event_id':range(1,7),'known':[False,True,True,True,True,True],
                          'success':[True]*6,'has_prior_success':[False,True,False,True,True,True]})
    alerts = pd.DataFrame({'event_id':[1,2,3,4,6],'timestamp':times[[0,1,2,3,5]],
                           'user_id':['u','u','v','u','v'],'ip_address':['ip']*5,
                           'rule_names':['']*5,'model_alert':[True]*5,'score':[.9]*5})
    return truth,alerts


def test_public_cases_budget_covers_only_alerted_members_and_opening_history():
    from src.rba_triage import evaluate_alerts
    truth,alerts = fixture_stream()
    result = evaluate_alerts(truth,alerts,'candidate',budgets=(1,None))
    summary,groups,daily,details,samples = result
    small = summary[summary.daily_budget=='1'].iloc[0]
    assert small.cases==3 and small.reviewed_cases==2 and small.overflow_cases==1
    assert small.alert_events==5 and small.repeated_alerts==2
    assert small.known_labels==5 and small.known_alerted==4 and small.known_reviewed==3
    assert small.known_record_coverage==pytest.approx(3/5)
    selected = details[details.daily_budget=='1'].set_index('event_id')
    assert not selected.loc[4,'reviewed'] and not selected.loc[5,'reviewed']
    assert selected.loc[2,'reviewed'] and selected.loc[6,'reviewed']
    group = groups[groups.daily_budget=='1'].set_index('group')
    assert group.loc['no_prior_success','cases']==2
    assert group.loc['has_prior_success','cases']==1
    assert group.loc['has_prior_success','overflow_cases']==1
    assert summary.iloc[1].known_reviewed==4
    assert (daily[daily.daily_budget=='1'].reviewed_cases<=1).all()
    assert not {'user_id','ip_address','known'} & set(samples)
    assert not {'fpr','precision','f1','episodes'} & set(summary)


def test_labels_cannot_change_grouping_admission_or_burden():
    from src.rba_triage import evaluate_alerts
    truth,alerts = fixture_stream()
    first = evaluate_alerts(truth,alerts,'candidate',budgets=(1,))
    second = evaluate_alerts(truth.assign(known=False),alerts,'candidate',budgets=(1,))
    pd.testing.assert_frame_equal(first[2],second[2])
    pd.testing.assert_frame_equal(first[4],second[4])
    for column in ['alert_events','cases','reviewed_cases','overflow_cases','repeated_alerts']:
        assert first[0][column].tolist()==second[0][column].tolist()
    assert pd.isna(second[0].known_record_coverage.iloc[0])
    with pytest.raises(ValueError,match='labels'):
        group_alerts(alerts.assign(known=True))


def test_no_public_alerts_and_failed_known_record_are_preserved():
    from src.rba_triage import evaluate_alerts
    truth,alerts = fixture_stream()
    truth.loc[4,'success']=False
    summary,groups,_,details,_ = evaluate_alerts(truth,alerts.iloc[:0],'candidate',budgets=(0,))
    assert summary.iloc[0].known_labels==5 and summary.iloc[0].known_success_labels==4
    assert summary.iloc[0].known_reviewed==0 and summary.iloc[0].cases==0
    assert len(details)==5 and not details.reviewed.any()
    assert groups.cases.sum()==0


def test_streaming_observer_matches_default_replay(tmp_path):
    from zipfile import ZipFile
    from src.rba_context import replay
    n=100
    raw=pd.DataFrame({'Login Timestamp':pd.date_range('2020-01-01',periods=n,freq='1s').astype(str),
                       'User ID':['u']*n,'IP Address':['192.0.2.1']*n,'Login Successful':['True']*n,
                       'Is Account Takeover':['False']*n,'ASN':['1']*n,'Country':['US']*n,
                       'OS Name and Version':['OS']*n,'Browser Name and Version':['Browser']*n,
                       'Device Type':['desktop']*n})
    archive=tmp_path/'tiny.zip'
    with ZipFile(archive,'w') as z:
        z.writestr('rba-dataset.csv',raw.to_csv(index=False))
    reference,info = replay(archive,100,20,40,60,13)
    observed=[]
    def observer(events,part):
        assert events.source_row.tolist()==part['source_row'].tolist()
        observed.append({k:v.copy() for k,v in part.items()})
    empty,again = replay(archive,100,20,40,60,13,observer=observer,retain_scores=False)
    assert empty=={} and again==info
    for key in reference:
        np.testing.assert_array_equal(reference[key],np.concatenate([p[key] for p in observed]))


def test_small_public_study_publishes_consistent_redacted_outputs(tmp_path,monkeypatch):
    import json
    from src import rba_triage
    from src.experiment_runs import resolve_run
    truth,small_alerts=fixture_stream()
    times=pd.to_datetime(['2020-01-01 23:55','2020-01-02 00:01','2020-01-02 00:02',
                          '2020-01-02 00:05','2020-01-02 00:06','2020-01-02 00:07'])
    event_ids=np.arange(8_000_001,8_000_007)
    successes=np.array([True,True,True,True,False,True])
    scores=np.array([.9,.9,.9,.9,.1,.9])
    events=pd.DataFrame({'source_row':event_ids,'timestamp':times,
                          'user_id':['u','u','v','u','u','v'],'ip_address':['ip']*6})
    part={'source_row':event_ids,'time':times.as_unit('ns').asi8,'success':successes,
          'known':truth.known.to_numpy(),'prior':truth.has_prior_success.to_numpy(dtype=int),
          'forest':scores,'global_surprisal':scores,'surprisal_a10':scores}
    lock={'archive_sha256':'archive','selected':'surprisal_a10',
          'thresholds':{name:.5 for name in rba_triage.METHODS}}
    parent=tmp_path/'parent.json'
    parent.write_text(json.dumps(lock))
    reference=tmp_path/'reference.csv'
    pd.DataFrame([{'model':name,'events':6,'alerts':5,'known_positives':5,'known_positives_alerted':4}
                  for name in rba_triage.METHODS]).to_csv(reference,index=False)
    monkeypatch.setattr(rba_triage,'archive_hashes',lambda _:('archive',rba_triage.PUBLISHER_MD5))
    def replay(*args,**kwargs):
        assert kwargs['retain_scores'] is False and kwargs['extra_boundaries']==(8_000_000,)
        kwargs['observer'](events,part)
        return {},{'source_rows':12_000_000}
    monkeypatch.setattr(rba_triage,'replay',replay)
    summary,audit=rba_triage.run(root=tmp_path,selection_path=parent,reference_path=reference)
    output=resolve_run(tmp_path/'reports/rba_triage')
    assert len(summary)==16 and all(audit['event_reference_checks'].values())
    assert (summary.known_reviewed==4).all()
    assert audit['known_failed_labels']==1
    details=pd.read_csv(output/'known_label_review.csv')
    assert len(details)==80 and not {'user_id','ip_address'} & set(details)
    for file in output.glob('*.csv'):
        assert not {'user_id','ip_address'} & set(pd.read_csv(file))
    old_pointer=(tmp_path/'reports/rba_triage/latest_complete.json').read_bytes()
    pd.DataFrame([{'model':name,'events':6,'alerts':99,'known_positives':5,'known_positives_alerted':4}
                  for name in rba_triage.METHODS]).to_csv(reference,index=False)
    with pytest.raises(ValueError,match='Event counts differ'):
        rba_triage.run(root=tmp_path,selection_path=parent,reference_path=reference)
    assert (tmp_path/'reports/rba_triage/latest_complete.json').read_bytes()==old_pointer
    assert resolve_run(tmp_path/'reports/rba_triage')==output
