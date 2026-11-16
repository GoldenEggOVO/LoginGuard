import pandas as pd
import pytest

from src.triage import group_alerts, evaluate_cases
from src.triage_study import fit_reference, make_alerts, run_study
from src.generate_data import generate


def test_cases_are_causal_fixed_windows_and_daily_budget():
    alerts = pd.DataFrame([
        (1, '2024-01-01 23:55:00', 'u', 'ip', 'repeated_failures', False, .3),
        (2, '2024-01-02 00:01:00', 'u', 'ip', '', True, .9),
        (3, '2024-01-02 00:02:00', 'v', 'ip', '', True, .8),
        (4, '2024-01-02 00:05:00', 'u', 'ip', '', True, .9),
    ], columns=['event_id', 'timestamp', 'user_id', 'ip_address', 'rule_names', 'model_alert', 'score'])
    cases, members = group_alerts(alerts, daily_budget=1)
    assert members.case_id.tolist() == [1, 1, 2, 3]
    assert cases.admitted.tolist() == [True, True, False]
    assert cases.alert_count.tolist() == [2, 1, 1]
    prefix_cases, prefix_members = group_alerts(alerts.iloc[:2], daily_budget=1)
    assert prefix_members.case_id.tolist() == members.case_id.iloc[:2].tolist()
    assert prefix_cases.admitted.tolist() == cases.admitted.iloc[:1].tolist()
    with pytest.raises(ValueError, match='labels'):
        group_alerts(alerts.assign(is_anomaly=1), daily_budget=1)
    with pytest.raises(ValueError):
        group_alerts(alerts.iloc[::-1], daily_budget=1)


def test_evaluation_only_counts_alerted_events_in_admitted_cases():
    events = pd.DataFrame([(1, '2024-01-01 00:00:00'), (2, '2024-01-01 00:01:00'),
                           (3, '2024-01-01 00:02:00'), (4, '2024-01-01 00:03:00')],
                          columns=['event_id', 'timestamp'])
    episodes = pd.DataFrame({'event_id':[1,2,3,4], 'episode_id':['a','a','b',None]})
    alerts = pd.DataFrame([(2, '2024-01-01 00:01:00', 'u', 'ip', '', True, .9),
                           (3, '2024-01-01 00:02:00', 'v', 'ip', '', True, .9),
                           (4, '2024-01-01 00:03:00', 'u', 'ip', '', True, .9)],
                          columns=['event_id','timestamp','user_id','ip_address','rule_names','model_alert','score'])
    cases, members = group_alerts(alerts, daily_budget=1)
    summary, detail = evaluate_cases(events, episodes, cases, members)
    assert summary['episodes'] == 2 and summary['episodes_covered'] == 1
    assert summary['reviewed_cases'] == 1 and summary['overflow_cases'] == 1
    assert summary['repeated_alerts'] == 1
    assert detail.set_index('episode_id').loc['a','latency_seconds'] == 60
    assert pd.isna(detail.set_index('episode_id').loc['b','latency_seconds'])


def test_preexisting_normal_case_does_not_detect_unalerted_attack():
    events = pd.DataFrame([(1, '2024-01-01 00:00:00'), (2, '2024-01-01 00:01:00'),
                           (3, '2024-01-01 00:02:00')], columns=['event_id', 'timestamp'])
    episodes = pd.DataFrame({'event_id': [1, 2, 3], 'episode_id': [None, 'a', 'a']})
    alerts = pd.DataFrame([(1, '2024-01-01 00:00:00', 'u', 'ip', '', True, .7)],
                          columns=['event_id','timestamp','user_id','ip_address','rule_names','model_alert','score'])
    cases, members = group_alerts(alerts, daily_budget=1)
    summary, _ = evaluate_cases(events, episodes, cases, members)
    assert summary['episodes_covered'] == 0
    later = pd.DataFrame([(3, '2024-01-01 00:02:00', 'u', 'ip', '', True, .8)], columns=alerts.columns)
    cases, members = group_alerts(pd.concat([alerts, later], ignore_index=True), daily_budget=1)
    summary, detail = evaluate_cases(events, episodes, cases, members)
    assert summary['episodes_covered'] == 1
    assert detail.iloc[0].latency_seconds == 60


def test_no_alerts_produces_uncovered_batch_without_crashing():
    events = pd.DataFrame([(1, '2024-01-01 00:00:00')], columns=['event_id', 'timestamp'])
    episodes = pd.DataFrame({'event_id': [1], 'episode_id': ['a']})
    alerts = pd.DataFrame(columns=['event_id','timestamp','user_id','ip_address','rule_names','model_alert','score'])
    cases, members = group_alerts(alerts, daily_budget=0)
    summary, details = evaluate_cases(events, episodes, cases, members)
    assert summary['episodes_missed'] == 1 and summary['reviewed_cases'] == 0
    assert pd.isna(details.iloc[0].latency_seconds)


def test_no_attack_batches_returns_undefined_recall():
    events = pd.DataFrame([(1, '2024-01-01 00:00:00')], columns=['event_id', 'timestamp'])
    episodes = pd.DataFrame({'event_id': [1], 'episode_id': [None]})
    alerts = pd.DataFrame(columns=['event_id','timestamp','user_id','ip_address','rule_names','model_alert','score'])
    cases, members = group_alerts(alerts)
    summary, _ = evaluate_cases(events, episodes, cases, members)
    assert summary['episodes'] == 0 and pd.isna(summary['episode_recall'])


def test_triage_study_writes_label_free_local_display(tmp_path):
    results = run_study(seeds=(241,), budgets=(2, None), root=tmp_path)
    assert len(results) == 6
    assert set(results.method) == {'Rules', 'Isolation Forest', 'Union'}
    assert (results.cases == results.reviewed_cases + results.overflow_cases).all()
    assert (results.episodes == results.episodes_covered + results.episodes_missed).all()
    cases = pd.read_csv(tmp_path / 'reports/triage/example_cases.csv')
    assert not {'is_anomaly', 'attack_type', 'episode_id'} & set(cases.columns)
    assert (tmp_path / 'reports/triage/example.html').exists()
    page = (tmp_path / 'reports/triage/example.html').read_text(encoding='utf-8')
    assert 'Synthetic alert triage' in page and 'Admitted' in page
    assert 'episode_id' not in page and 'attack_type' not in page
    assert '<script' not in page.lower() and 'https://' not in page.lower()
    for _, group in results.groupby('method'):
        small = group[group.daily_budget == '2'].episode_recall.iloc[0]
        unlimited = group[group.daily_budget == 'unlimited'].episode_recall.iloc[0]
        assert small <= unlimited


def test_union_has_one_alert_for_each_event_detected_by_either_method():
    events, _ = generate(241)
    model, cutoff = fit_reference()
    rule, _ = make_alerts(events, model, cutoff, 'Rules')
    forest, _ = make_alerts(events, model, cutoff, 'Isolation Forest')
    union, _ = make_alerts(events, model, cutoff, 'Union')
    assert union.event_id.is_unique
    assert set(union.event_id) == set(rule.event_id) | set(forest.event_id)
