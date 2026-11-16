from collections import deque
from functools import lru_cache
import hashlib
import ipaddress
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = ['success', 'account_failures_60s', 'ip_failures_60s', 'ip_distinct_users_10m', 'prior_consecutive_failures']
RAW_COLUMNS = ['Login Timestamp', 'User ID', 'IP Address', 'Login Successful', 'Is Account Takeover']
PUBLISHER_MD5 = 'cc1b1078b3929650e6c08678caffcc57'


@lru_cache(maxsize=200_000)
def valid_ip(value):
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def adapt_rows(raw):
    timestamp = pd.to_datetime(raw['Login Timestamp'], errors='coerce', format='mixed')
    user = raw['User ID'].astype('string').str.strip()
    source = raw['IP Address'].astype('string').str.strip()
    outcome = raw['Login Successful'].astype('string').str.strip().str.lower().map({'true': 1, 'false': 0})
    label = raw['Is Account Takeover'].astype('string').str.strip().str.lower()
    user_invalid = user.isna() | user.isin(['', '?', '-'])
    ip_invalid = ~source.fillna('').map(valid_ip)
    valid = timestamp.notna() & ~user_invalid & ~ip_invalid & outcome.notna()
    states = np.select([label.eq('true').fillna(False), label.eq('false').fillna(False)],
                       ['known_positive', 'unlabelled_explicit_false'], default='unlabelled_missing_or_invalid')
    audit = {'source_rows': len(raw), 'invalid_timestamp': int(timestamp.isna().sum()),
             'missing_account': int(user_invalid.sum()), 'invalid_source_ip': int(ip_invalid.sum()),
             'invalid_login_status': int(outcome.isna().sum()), 'excluded_rows': int((~valid).sum()),
             'excluded_known_positive': int(((~valid).to_numpy() & (states == 'known_positive')).sum())}
    events = pd.DataFrame({'source_row': np.arange(1, len(raw) + 1), 'timestamp': timestamp,
                           'user_id': user, 'ip_address': source, 'success': outcome})[valid].reset_index(drop=True)
    events['success'] = events.success.astype('int8')
    labels = pd.DataFrame({'source_row': np.arange(1, len(raw) + 1), 'label_state': states})[valid].reset_index(drop=True)
    return events, labels, audit


def chronological_features(events, state=None):
    times = pd.to_datetime(events.timestamp)
    if times.isna().any() or not times.is_monotonic_increasing:
        raise ValueError('Events must have valid chronological timestamps')
    state = {} if state is None else state
    if len(times) and 'last_timestamp' in state and times.iloc[0] < state['last_timestamp']:
        raise ValueError('Events must have valid chronological timestamps across chunks')
    ns = times.to_numpy(dtype='datetime64[ns]').astype('int64')
    failed, attempted = state.setdefault('failed', deque()), state.setdefault('attempted', deque())
    account_fail, ip_fail = state.setdefault('account_fail', {}), state.setdefault('ip_fail', {})
    per_ip_users, streak = state.setdefault('per_ip_users', {}), state.setdefault('streak', {})
    values = np.empty((len(events), len(FEATURES)), dtype=np.int32)
    for i, (now, user, ip, success) in enumerate(zip(ns, events.user_id, events.ip_address, events.success)):
        while failed and now - failed[0][0] > 60_000_000_000:
            _, old_user, old_ip = failed.popleft()
            account_fail[old_user] -= 1
            ip_fail[old_ip] -= 1
            if not account_fail[old_user]:
                del account_fail[old_user]
            if not ip_fail[old_ip]:
                del ip_fail[old_ip]
        while attempted and now - attempted[0][0] > 600_000_000_000:
            _, old_ip, old_user = attempted.popleft()
            users = per_ip_users[old_ip]
            users[old_user] -= 1
            if not users[old_user]:
                del users[old_user]
            if not users:
                del per_ip_users[old_ip]
        prior = streak.get(user, 0)
        if not success:
            failed.append((now, user, ip))
            account_fail[user] = account_fail.get(user, 0) + 1
            ip_fail[ip] = ip_fail.get(ip, 0) + 1
            streak[user] = prior + 1
        else:
            streak.pop(user, None)
        attempted.append((now, ip, user))
        users = per_ip_users.setdefault(ip, {})
        users[user] = users.get(user, 0) + 1
        values[i] = [success, account_fail.get(user, 0), ip_fail.get(ip, 0), len(users), prior]
    if len(times):
        state['last_timestamp'] = times.iloc[-1]
    return pd.DataFrame(values, columns=FEATURES)


def alert_budget_cutoff(scores, budget=.02):
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or not 0 <= budget < 1:
        raise ValueError('A non-empty finite score vector and valid budget are required')
    return float(np.nextafter(np.sort(scores)[::-1][int(np.floor(budget * len(scores)))], np.inf))


def archive_hashes(path):
    sha256, md5 = hashlib.sha256(), hashlib.md5()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha256.update(block)
            md5.update(block)
    return sha256.hexdigest(), md5.hexdigest()


def run(archive_path=Path('data/public/rba/rba-dataset.zip'), max_rows=2_000_000, root=Path('.'), expected_md5=PUBLISHER_MD5):
    output = Path(root) / 'reports' / 'rba'
    output.mkdir(parents=True, exist_ok=True)
    sha256, md5 = archive_hashes(archive_path)
    if expected_md5 is not None and md5.lower() != expected_md5.lower():
        raise ValueError('Archive MD5 does not match expected publisher checksum')
    with ZipFile(archive_path) as archive:
        member = archive.getinfo('rba-dataset.csv')
        with archive.open(member) as stream:
            raw = pd.read_csv(stream, usecols=RAW_COLUMNS, nrows=max_rows, dtype=str, keep_default_na=False)
    events, labels, audit = adapt_rows(raw)
    del raw
    if len(events) < 5 or not events.timestamp.is_monotonic_increasing:
        raise ValueError('Sample is too small or not chronological; no evaluation was performed')
    boundary_60 = events.timestamp.iloc[int(len(events) * .6)]
    boundary_80 = events.timestamp.iloc[int(len(events) * .8)]
    train = events.timestamp < boundary_60
    validation = (events.timestamp >= boundary_60) & (events.timestamp < boundary_80)
    test = events.timestamp >= boundary_80
    if not train.any() or not validation.any() or not test.any():
        raise ValueError('Timestamp split yielded an empty period')
    features = chronological_features(events)
    model = IsolationForest(n_estimators=200, contamination='auto', random_state=42, n_jobs=1)
    model.fit(features.loc[train, FEATURES])
    validation_scores = -model.score_samples(features.loc[validation, FEATURES])
    threshold = alert_budget_cutoff(validation_scores)
    test_features = features.loc[test].reset_index(drop=True)
    scores = -model.score_samples(test_features[FEATURES])
    model_alert = scores >= threshold
    rule_flags = pd.DataFrame({'repeated_failures': test_features.account_failures_60s >= 5,
                               'multiple_accounts_per_ip': test_features.ip_distinct_users_10m >= 5,
                               'success_after_failures': (test_features.success == 1) & (test_features.prior_consecutive_failures >= 3)})
    rules = rule_flags.any(axis=1).to_numpy()
    test_labels = labels.loc[test, 'label_state'].to_numpy()
    known = test_labels == 'known_positive'
    top_train_counts = events.loc[train, 'user_id'].value_counts()
    dominant_id = top_train_counts.index[0]
    test_events = events.loc[test].reset_index(drop=True)
    dominant_test = test_events.user_id.eq(dominant_id).to_numpy()
    rows, sensitivity_rows = [], []
    for name, alerts in [('Rules', rules), ('Isolation Forest', model_alert)]:
        for mask, target in [(np.ones(len(test_labels), dtype=bool), rows), (~dominant_test, sensitivity_rows)]:
            selected_alerts, selected_known = alerts[mask], known[mask]
            target.append({'method': name, 'test_events': int(mask.sum()), 'alerts': int(selected_alerts.sum()),
                           'alert_rate': float(selected_alerts.mean()) if mask.any() else None,
                           'known_positives': int(selected_known.sum()),
                           'known_positives_alerted': int(selected_alerts[selected_known].sum()),
                           'known_positive_recall': float(selected_alerts[selected_known].mean()) if selected_known.any() else None,
                           'unlabelled_events': int((~selected_known).sum()),
                           'unlabelled_alerts': int(selected_alerts[~selected_known].sum()),
                           'unlabelled_alert_rate': float(selected_alerts[~selected_known].mean()) if (~selected_known).any() else None})
    result = pd.DataFrame(rows)
    result.to_csv(output / 'metrics.csv', index=False)
    pd.DataFrame(sensitivity_rows).to_csv(output / 'metrics_without_train_top_id.csv', index=False)
    identity_audit = {'selection': 'most frequent User ID in training; post-hoc sensitivity only',
                      'train_count': int(top_train_counts.iloc[0]), 'test_count': int(dominant_test.sum()),
                      'test_successes': int(test_events.loc[dominant_test, 'success'].sum()),
                      'test_unique_ips': int(test_events.loc[dominant_test, 'ip_address'].nunique()),
                      'test_rule_alerts': int(rules[dominant_test].sum()),
                      'test_repeated_failures_rule': int(rule_flags.repeated_failures.to_numpy()[dominant_test].sum()),
                      'test_model_alerts': int(model_alert[dominant_test].sum()),
                      'note': 'Identity semantics are unknown; no rows removed from training, calibration, or feature history.'}
    (output / 'identity_audit.json').write_text(json.dumps(identity_audit, indent=2), encoding='utf-8')
    pd.DataFrame([{'rule': name, 'test_events_triggered': int(flag.sum()),
                   'known_positives_triggered': int(flag.to_numpy()[known].sum())}
                  for name, flag in rule_flags.items()]).to_csv(output / 'rule_breakdown.csv', index=False)
    known_rows = pd.DataFrame({'source_row': events.loc[test, 'source_row'].to_numpy()[known],
                               'model_score': scores[known], 'model_alert': model_alert[known], 'rule_alert': rules[known]})
    known_rows = pd.concat([known_rows.reset_index(drop=True), test_features.loc[known].reset_index(drop=True)], axis=1)
    known_rows.to_csv(output / 'known_positive_rows.csv', index=False)
    audit.update({'archive_sha256': sha256, 'archive_md5': md5,
                  'publisher_md5_verified': expected_md5 is not None, 'requested_prefix_rows': max_rows,
                  'valid_rows': len(events), 'feature_columns': FEATURES, 'train_rows': int(train.sum()),
                  'validation_rows': int(validation.sum()), 'test_rows': int(test.sum()),
                  'first_timestamp': str(events.timestamp.iloc[0]), 'last_timestamp': str(events.timestamp.iloc[-1]),
                  'validation_start': str(boundary_60), 'test_start': str(boundary_80),
                  'known_positives_by_split': {name: int(labels.loc[mask, 'label_state'].eq('known_positive').sum())
                                               for name, mask in [('train', train), ('validation', validation), ('test', test)]},
                  'label_states': labels.label_state.value_counts().to_dict(), 'threshold': threshold,
                  'validation_alert_rate_all_events': float((validation_scores >= threshold).mean())})
    (output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))
    print(result.to_string(index=False))
    return result, audit


if __name__ == '__main__':
    run()
