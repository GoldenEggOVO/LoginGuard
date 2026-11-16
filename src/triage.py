from numbers import Integral
import numpy as np
import pandas as pd

CASE_COLUMNS = ['case_id', 'opened_at', 'last_alert_at', 'user_id', 'ip_address', 'first_event_id',
                'alert_count', 'rules', 'model_alert_seen', 'max_score', 'admitted']
MEMBER_COLUMNS = ['event_id', 'case_id', 'timestamp']


def group_alerts(alerts, daily_budget=5, window_seconds=600):
    if {'is_anomaly', 'attack_type', 'episode_id', 'episode_step', 'known', 'label_state',
            'Is Account Takeover', 'Is Attack IP'} & set(alerts.columns):
        raise ValueError('Evaluation labels must not be included in alerts')
    if (daily_budget is not None and (isinstance(daily_budget, bool) or not isinstance(daily_budget, Integral) or daily_budget < 0)):
        raise ValueError('daily_budget must be a nonnegative integer or None')
    if window_seconds <= 0:
        raise ValueError('window_seconds must be positive')
    times = pd.to_datetime(alerts.timestamp)
    if not times.is_monotonic_increasing or not alerts.event_id.is_unique:
        raise ValueError('Alerts must be chronological and event IDs unique')
    cases, members, active, used = [], [], {}, {}
    for row, ts in zip(alerts.itertuples(index=False), times):
        key = (row.user_id, row.ip_address)
        case = active.get(key)
        if case is None or ts >= case['opened_at'] + pd.Timedelta(seconds=window_seconds):
            day = ts.date()
            count = used.get(day, 0)
            admitted = daily_budget is None or count < daily_budget
            if admitted:
                used[day] = count + 1
            case = {'case_id': len(cases) + 1, 'opened_at': ts, 'last_alert_at': ts, 'user_id': row.user_id,
                    'ip_address': row.ip_address, 'first_event_id': row.event_id, 'alert_count': 0,
                    'rules': set(), 'model_alert_seen': False, 'max_score': np.nan, 'admitted': admitted}
            active[key] = case
            cases.append(case)
        case['last_alert_at'] = ts
        case['alert_count'] += 1
        if pd.notna(row.rule_names) and row.rule_names:
            case['rules'].update(row.rule_names.split(';'))
        case['model_alert_seen'] |= bool(row.model_alert)
        if pd.notna(row.score):
            case['max_score'] = max(case['max_score'], row.score) if pd.notna(case['max_score']) else row.score
        members.append({'event_id': row.event_id, 'case_id': case['case_id'], 'timestamp': ts})
    for case in cases:
        case['rules'] = ';'.join(sorted(case['rules']))
    return pd.DataFrame(cases, columns=CASE_COLUMNS), pd.DataFrame(members, columns=MEMBER_COLUMNS)


def evaluate_cases(events, episodes, cases, members):
    if not events.event_id.equals(episodes.event_id):
        raise ValueError('Events and synthetic episode metadata must align')
    truth = events[['event_id', 'timestamp']].merge(episodes[['event_id', 'episode_id']], on='event_id', validate='one_to_one')
    truth['timestamp'] = pd.to_datetime(truth.timestamp)
    starts = truth[truth.episode_id.notna()].groupby('episode_id').timestamp.min()
    reviewed = cases.loc[cases.admitted, 'case_id'] if len(cases) else pd.Series(dtype=int)
    admitted_members = members[members.case_id.isin(reviewed)]
    hits = admitted_members[['event_id', 'case_id']].merge(truth, on='event_id', validate='one_to_one')
    hits = hits[hits.episode_id.notna()]
    first_hits = hits.groupby('episode_id').timestamp.min()
    details = pd.DataFrame({'episode_id': starts.index, 'started_at': starts.values})
    details = details.merge(first_hits.rename('first_alert_at').reset_index(), on='episode_id', how='left', validate='one_to_one')
    details['first_alert_at'] = pd.to_datetime(details.first_alert_at)
    details['covered'] = details.first_alert_at.notna()
    details['latency_seconds'] = (details.first_alert_at - details.started_at).dt.total_seconds()
    n_cases, n_reviewed, n_episodes = len(cases), len(reviewed), len(details)
    n_covered = int(details.covered.sum())
    summary = {'alert_events': len(members), 'cases': n_cases, 'reviewed_cases': n_reviewed,
               'overflow_cases': n_cases - n_reviewed, 'repeated_alerts': len(members) - n_cases,
               'episodes': n_episodes, 'episodes_covered': n_covered, 'episodes_missed': n_episodes - n_covered,
               'episode_recall': n_covered / n_episodes if n_episodes else np.nan,
               'label_relative_case_precision': hits.case_id.nunique() / n_reviewed if n_reviewed else np.nan,
               'median_covered_latency_seconds': details.latency_seconds.median() if n_covered else np.nan}
    return summary, details
