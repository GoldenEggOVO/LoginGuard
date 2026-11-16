import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .rba_validation import (FEATURES, PUBLISHER_MD5, RAW_COLUMNS, adapt_rows,
                             alert_budget_cutoff, archive_hashes, chronological_features)

HISTORY_COLUMNS = ['has_prior_success', 'ip_seen_on_prior_success', 'has_prior_known_type',
                   'type_seen_on_prior_success', 'current_type_available']
KNOWN_TYPES = {'mobile': 1, 'desktop': 2, 'tablet': 3, 'bot': 4}
DEVELOPMENT_MISSES = (1799492, 1845944, 1851695, 1908328)


def history_features(events):
    success_count, ip_history, type_masks = {}, set(), {}
    values = np.empty((len(events), 6), dtype=np.int32)
    for i, (user, ip, success, raw_type) in enumerate(zip(events.user_id, events.ip_address, events.success, events.device_type)):
        kind = KNOWN_TYPES.get(str(raw_type).strip().lower(), 0) if pd.notna(raw_type) else 0
        count = success_count.get(user, 0)
        type_mask = type_masks.get(user, 0)
        seen_ip = (user, ip) in ip_history
        seen_type = bool(kind and type_mask & (1 << kind))
        values[i] = [int(count > 0), int(seen_ip), int(type_mask != 0), int(seen_type), int(kind != 0), count]
        if success:
            success_count[user] = count + 1
            ip_history.add((user, ip))
            if kind:
                type_masks[user] = type_mask | (1 << kind)
    return pd.DataFrame(values, columns=HISTORY_COLUMNS + ['prior_success_count'])


def summarize(alerts, labels, mask, name, threshold, validation_rate):
    selected_alerts, selected_labels = alerts[mask], labels[mask]
    known = selected_labels == 'known_positive'
    return {'model': name, 'threshold': threshold, 'validation_alert_rate': validation_rate,
            'test_events': int(mask.sum()), 'alerts': int(selected_alerts.sum()),
            'alert_rate': float(selected_alerts.mean()) if mask.any() else None,
            'known_positives': int(known.sum()), 'known_positives_alerted': int(selected_alerts[known].sum()),
            'known_positive_recall': float(selected_alerts[known].mean()) if known.any() else None,
            'unlabelled_events': int((~known).sum()), 'unlabelled_alerts': int(selected_alerts[~known].sum()),
            'unlabelled_alert_rate': float(selected_alerts[~known].mean()) if (~known).any() else None}


def run(archive_path=Path('data/public/rba/rba-dataset.zip'), max_rows=4_000_000,
        train_rows=1_200_000, development_rows=2_000_000, retrospective_rows=DEVELOPMENT_MISSES,
        root=Path('.'), expected_md5=PUBLISHER_MD5):
    output = Path(root) / 'reports' / 'rba_history'
    output.mkdir(parents=True, exist_ok=True)
    sha256, md5 = archive_hashes(archive_path)
    if expected_md5 is not None and md5.lower() != expected_md5.lower():
        raise ValueError('Archive MD5 does not match expected publisher checksum')
    with ZipFile(archive_path) as archive:
        with archive.open('rba-dataset.csv') as stream:
            raw = pd.read_csv(stream, usecols=RAW_COLUMNS + ['Device Type'], nrows=max_rows, dtype=str, keep_default_na=False)
    events, labels, audit = adapt_rows(raw)
    events['device_type'] = raw['Device Type'].iloc[events.source_row.to_numpy() - 1].reset_index(drop=True)
    del raw
    if len(events) <= development_rows or not 0 < train_rows < development_rows or not events.timestamp.is_monotonic_increasing:
        raise ValueError('Insufficient valid rows or input is not chronological')
    train_start = events.timestamp.iloc[train_rows]
    holdout_start = events.timestamp.iloc[development_rows]
    train = events.timestamp < train_start
    validation = (events.timestamp >= train_start) & (events.timestamp < holdout_start)
    test = events.timestamp >= holdout_start
    if not train.any() or not validation.any() or not test.any():
        raise ValueError('Timestamp split yielded an empty period')
    base = chronological_features(events)
    history = history_features(events)
    combined = pd.concat([base, history[HISTORY_COLUMNS]], axis=1)
    labels_test = labels.loc[test, 'label_state'].to_numpy()
    top_counts = events.loc[train, 'user_id'].value_counts()
    top_id = top_counts.index[0]
    test_top = events.loc[test, 'user_id'].eq(top_id).to_numpy(dtype=bool)
    retrospective = events.source_row.isin(retrospective_rows) & ~test
    retrospective_data = pd.concat([events.loc[retrospective, ['source_row']].reset_index(drop=True),
                                    base.loc[retrospective].reset_index(drop=True),
                                    history.loc[retrospective].reset_index(drop=True)], axis=1)
    known_holdout = labels_test == 'known_positive'
    positive_details = pd.concat([events.loc[test, ['source_row']].reset_index(drop=True).loc[known_holdout].reset_index(drop=True),
                                  base.loc[test].reset_index(drop=True).loc[known_holdout].reset_index(drop=True),
                                  history.loc[test].reset_index(drop=True).loc[known_holdout].reset_index(drop=True)], axis=1)
    history_seen_test = history.loc[test, 'has_prior_success'].to_numpy(dtype=bool)
    rows, sensitivity, segments = [], [], []
    for name, columns in [('baseline', FEATURES), ('history', FEATURES + HISTORY_COLUMNS)]:
        model = IsolationForest(n_estimators=200, contamination='auto', random_state=42, n_jobs=1)
        model.fit(combined.loc[train, columns])
        val_scores = -model.score_samples(combined.loc[validation, columns])
        threshold = alert_budget_cutoff(val_scores)
        val_rate = float((val_scores >= threshold).mean())
        test_scores = -model.score_samples(combined.loc[test, columns])
        alerts = test_scores >= threshold
        positive_details[name + '_score'] = test_scores[known_holdout]
        positive_details[name + '_alert'] = alerts[known_holdout]
        rows.append(summarize(alerts, labels_test, np.ones(len(alerts), dtype=bool), name, threshold, val_rate))
        sensitivity.append(summarize(alerts, labels_test, ~test_top, name, threshold, val_rate))
        for segment, mask in [('no_prior_success', ~history_seen_test), ('has_prior_success', history_seen_test)]:
            segments.append({'segment': segment, **summarize(alerts, labels_test, mask, name, threshold, val_rate)})
        retro_scores = -model.score_samples(combined.loc[retrospective, columns])
        retrospective_data[name + '_validation_score'] = retro_scores
        retrospective_data[name + '_validation_alert'] = retro_scores >= threshold
    result = pd.DataFrame(rows)
    result.to_csv(output / 'holdout_metrics.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(output / 'holdout_without_train_top_id.csv', index=False)
    pd.DataFrame(segments).to_csv(output / 'holdout_segments.csv', index=False)
    retrospective_data.to_csv(output / 'missed_case_history.csv', index=False)
    positive_details.to_csv(output / 'holdout_positive_details.csv', index=False)
    identity = {'selection': 'most frequent User ID in training', 'train_count': int(top_counts.iloc[0]),
                'train_successes': int(events.loc[train & events.user_id.eq(top_id), 'success'].sum()),
                'train_unique_ips': int(events.loc[train & events.user_id.eq(top_id), 'ip_address'].nunique()),
                'second_most_frequent_train_count': int(top_counts.iloc[1]) if len(top_counts) > 1 else None,
                'validation_count': int(events.loc[validation, 'user_id'].eq(top_id).sum()),
                'validation_successes': int(events.loc[validation & events.user_id.eq(top_id), 'success'].sum()),
                'validation_unique_ips': int(events.loc[validation & events.user_id.eq(top_id), 'ip_address'].nunique()),
                'holdout_count': int(test_top.sum()),
                'holdout_successes': int(events.loc[test & events.user_id.eq(top_id), 'success'].sum()),
                'holdout_unique_ips': int(events.loc[test & events.user_id.eq(top_id), 'ip_address'].nunique()),
                'note': 'Public source does not identify this pseudonym as a placeholder or a single real account.'}
    (output / 'identity_audit.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    audit.update({'archive_sha256': sha256, 'archive_md5': md5, 'publisher_md5_verified': expected_md5 is not None,
                  'valid_rows': len(events), 'train_rows': int(train.sum()), 'validation_rows': int(validation.sum()),
                  'holdout_rows': int(test.sum()), 'validation_start': str(train_start), 'holdout_start': str(holdout_start),
                  'last_timestamp': str(events.timestamp.iloc[-1]), 'history_columns': HISTORY_COLUMNS,
                  'known_positives_by_split': {name: int(labels.loc[mask, 'label_state'].eq('known_positive').sum())
                                               for name, mask in [('train', train), ('validation', validation), ('holdout', test)]},
                  'retrospective_rows_present': int(retrospective.sum())})
    (output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))
    print(result.to_string(index=False))
    return result, audit


if __name__ == '__main__':
    run()
