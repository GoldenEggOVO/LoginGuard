import argparse
from collections import Counter
import hashlib
import json
from math import log
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .rba_validation import (FEATURES, RAW_COLUMNS, PUBLISHER_MD5, adapt_rows,
                             archive_hashes, alert_budget_cutoff, chronological_features)
from .experiment_runs import experiment_run, selection_run

CONTEXT_COLUMNS = ['IP Address', 'ASN', 'Country', 'OS Name and Version',
                   'Browser Name and Version', 'Device Type']
ALPHAS = (1, 10, 100)
CANDIDATES = ['global_surprisal'] + [f'{kind}_a{alpha}' for kind in ('surprisal', 'ratio') for alpha in ALPHAS]


class ContextRisk:
    """Successful-login frequencies; current values are scored before updating."""

    def __init__(self):
        self.global_counts = [Counter() for _ in CONTEXT_COLUMNS]
        self.user_counts = [Counter() for _ in CONTEXT_COLUMNS]
        self.user_totals = [Counter() for _ in CONTEXT_COLUMNS]
        self.totals = [0] * len(CONTEXT_COLUMNS)
        self.successes = Counter()

    def score_update(self, user, values, success, *, include_fields=False):
        """Optionally expose the same per-field evidence for scoring controls."""
        result = dict.fromkeys(CANDIDATES, 0.0)
        prior = self.successes[user]
        valid = 0
        global_fields, personal_fields, available = [0.0] * 6, [0.0] * 6, [0] * 6
        for j, raw in enumerate(values):
            value = str(raw).strip() if pd.notna(raw) else ''
            if value.lower() in ('', 'unknown', 'nan', 'none', '?', '-'):
                continue
            valid += 1
            global_count = self.global_counts[j][value]
            global_p = (global_count + 1) / (self.totals[j] + len(self.global_counts[j]) + 1)
            global_score = -log(global_p)
            if include_fields:
                global_fields[j], available[j] = global_score, 1
            result['global_surprisal'] += global_score
            total = self.user_totals[j][user]
            count = self.user_counts[j][(user, value)]
            for alpha in ALPHAS:
                personal_p = (count + alpha * global_p) / (total + alpha)
                result[f'surprisal_a{alpha}'] -= log(personal_p)
                if include_fields and alpha == 10:
                    personal_fields[j] = -log(personal_p)
                result[f'ratio_a{alpha}'] += log(global_p / personal_p) if total else global_score
            if success:
                self.global_counts[j][value] += 1
                self.user_counts[j][(user, value)] += 1
                self.user_totals[j][user] += 1
                self.totals[j] += 1
        if valid:
            result = {k: v / valid for k, v in result.items()}
        if success:
            self.successes[user] += 1
        if include_fields:
            result.update(global_fields=global_fields, personal_fields=personal_fields,
                          available_fields=available)
        return result, prior


def replay(archive_path, max_rows, train_rows, calibration_rows, development_rows, chunksize, extra_boundaries=(),
           observer=None, retain_scores=True):

    context, window, top_counts = ContextRisk(), {}, Counter()
    train_parts, parts, model, top_id = [], [], None, None
    offset, previous, excluded, excluded_known = 0, None, 0, 0
    boundaries = tuple(sorted(set((train_rows, calibration_rows, development_rows) + tuple(extra_boundaries))))
    previous_partition = 0
    split_times = {}
    scored_rows = 0
    with ZipFile(archive_path) as archive, archive.open('rba-dataset.csv') as stream:
        reader = pd.read_csv(stream, usecols=list(dict.fromkeys(RAW_COLUMNS + CONTEXT_COLUMNS)),
                             dtype=str, keep_default_na=False, chunksize=chunksize, nrows=max_rows)
        for chunk in reader:
            cuts = [0] + [x - offset for x in boundaries
                          if offset < x < offset + len(chunk)] + [len(chunk)]
            for left, right in zip(cuts, cuts[1:]):
                raw = chunk.iloc[left:right].reset_index(drop=True)
                start_row = offset + left
                events, labels, audit = adapt_rows(raw)
                excluded += audit['excluded_rows']
                excluded_known += audit['excluded_known_positive']
                values = raw[CONTEXT_COLUMNS].iloc[events.source_row.to_numpy() - 1].to_numpy()
                events['source_row'] += start_row
                if len(events) == 0:
                    continue
                if previous is not None and events.timestamp.iloc[0] < previous:
                    raise ValueError('Input must be chronological across chunks')
                partition = sum(start_row >= boundary for boundary in boundaries)
                if partition > previous_partition:
                    if previous is not None and events.timestamp.iloc[0] == previous:
                        raise ValueError('A split boundary cuts equal timestamps; revise protocol before evaluation')
                    for boundary in boundaries[previous_partition:partition]:
                        split_times[str(boundary)] = str(events.timestamp.iloc[0])
                previous_partition = partition
                previous = events.timestamp.iloc[-1]
                base = chronological_features(events, window)
                numeric = np.empty((len(events), len(CANDIDATES)), dtype=np.float64)
                priors = np.empty(len(events), dtype=np.int32)
                for i, (user, success) in enumerate(zip(events.user_id, events.success)):
                    scores, priors[i] = context.score_update(user, values[i], bool(success))
                    numeric[i] = [scores[k] for k in CANDIDATES]
                if start_row < train_rows:
                    train_parts.append(base.to_numpy())
                    top_counts.update(events.user_id)
                else:
                    if model is None:
                        if not train_parts:
                            raise ValueError('Training period is empty')
                        model = IsolationForest(n_estimators=200, contamination='auto', random_state=42, n_jobs=1)
                        model.fit(np.concatenate(train_parts))
                        train_parts.clear()
                        top_id = top_counts.most_common(1)[0][0]
                    part = {name: numeric[:, i] for i, name in enumerate(CANDIDATES)}
                    part.update(forest=-model.score_samples(base.to_numpy()),
                                source_row=events.source_row.to_numpy(),
                                time=events.timestamp.to_numpy(dtype='datetime64[ns]').astype('int64'),
                                success=events.success.to_numpy(dtype=bool),
                                known=labels.label_state.eq('known_positive').to_numpy(dtype=bool),
                                prior=priors, top=events.user_id.eq(top_id).to_numpy(dtype=bool))
                    scored_rows += len(events)
                    if observer is not None:
                        observer(events,part)
                    if retain_scores:
                        parts.append(part)
            offset += len(chunk)
            print(f'Replayed {offset:,} rows', flush=True)
    if offset < max_rows or not scored_rows:
        raise ValueError('Archive has insufficient rows for the frozen protocol')
    data = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]} if retain_scores else {}
    info = {'source_rows': offset, 'excluded_rows': excluded, 'excluded_known_positive': excluded_known,
            'split_timestamps': split_times, 'last_timestamp': str(previous),
            'top_id_train_count': top_counts.most_common(1)[0][1]}
    return data, info


def predictions(data, name, threshold):
    scores = data['forest'] if name.startswith('forest_') else data[name]
    return (scores >= threshold) & (True if name == 'forest_all' else data['success'])


def metrics(data, mask, name, threshold):
    success, known = data['success'][mask], data['known'][mask]
    alerts = predictions(data, name, threshold)[mask]
    unlabelled = ~known
    success_unlabelled = success & unlabelled
    return {'model': name, 'threshold': threshold, 'events': int(mask.sum()),
            'success_events': int(success.sum()), 'alerts': int(alerts.sum()),
            'failure_alerts': int(alerts[~success].sum()), 'known_positives': int(known.sum()),
            'known_positives_alerted': int(alerts[known].sum()),
            'known_positive_recall': float(alerts[known].mean()) if known.any() else None,
            'unlabelled_events': int(unlabelled.sum()), 'unlabelled_alerts': int(alerts[unlabelled].sum()),
            'unlabelled_alert_rate': float(alerts[unlabelled].mean()) if unlabelled.any() else None,
            'unlabelled_success_events': int(success_unlabelled.sum()),
            'unlabelled_success_alerts': int(alerts[success_unlabelled].sum()),
            'unlabelled_success_alert_rate': float(alerts[success_unlabelled].mean()) if success_unlabelled.any() else None}


def develop(data, train_rows, calibration_rows, development_rows):
    calibration = data['source_row'] <= calibration_rows
    development = (data['source_row'] > calibration_rows) & (data['source_row'] <= development_rows)
    thresholds, rates = {}, {}
    for name in ['forest_all','forest_success'] + CANDIDATES:
        scores = data['forest'] if name.startswith('forest_') else data[name]
        eligible = calibration if name == 'forest_all' else calibration & data['success']
        if not eligible.any():
            raise ValueError('No eligible calibration records')
        thresholds[name] = alert_budget_cutoff(scores[eligible])
        rates[name] = float((scores[eligible] >= thresholds[name]).mean())
    rows = [metrics(data, development, name, thresholds[name]) for name in CANDIDATES]
    selected = sorted(rows, key=lambda r: (-r['known_positives_alerted'], r['unlabelled_alerts'], r['model']))[0]['model']
    return {'selected':selected, 'thresholds':thresholds, 'calibration_eligible_alert_rates':rates,
            'development_metrics':rows, 'train_rows':train_rows, 'calibration_rows':calibration_rows,
            'development_rows':development_rows, 'alphas':list(ALPHAS), 'context_columns':CONTEXT_COLUMNS}


def run(archive_path=Path('data/public/rba/rba-dataset.zip'), max_rows=8_000_000,
        train_rows=1_200_000, calibration_rows=2_000_000, development_rows=4_000_000,
        chunksize=100_000, root=Path('.'), expected_md5=PUBLISHER_MD5, phase='both'):
    base = Path(root)/'reports'/'rba_context'
    source = selection_run(base) if phase=='holdout' else None
    with experiment_run(base,'development' if phase=='development' else 'holdout') as output:
        if source is not None:
            (output/'selection.json').write_bytes((source/'selection.json').read_bytes())
            if (source/'development_metrics.csv').exists():
                (output/'development_metrics.csv').write_bytes((source/'development_metrics.csv').read_bytes())
        return _run(archive_path,max_rows,train_rows,calibration_rows,development_rows,
                    chunksize,output,expected_md5,phase)


def _run(archive_path, max_rows, train_rows, calibration_rows, development_rows,
         chunksize, output, expected_md5, phase):
    if not 0 < train_rows < calibration_rows < development_rows < max_rows:
        raise ValueError('Invalid frozen split sizes')
    sha256, md5 = archive_hashes(archive_path)
    if expected_md5 is not None and md5.lower() != expected_md5.lower():
        raise ValueError('Archive MD5 does not match expected publisher checksum')
    selection_path = output/'selection.json'
    if phase in ('both','development'):
        data, info = replay(archive_path, development_rows, train_rows, calibration_rows, development_rows, chunksize)
        selection = develop(data, train_rows, calibration_rows, development_rows)
        selection.update(archive_sha256=sha256, publisher_md5_verified=expected_md5 is not None,
                         max_rows=max_rows, development_audit=info)
        selection_path.write_text(json.dumps(selection, indent=2), encoding='utf-8')
        pd.DataFrame(selection['development_metrics']).to_csv(output/'development_metrics.csv',index=False)
        del data
        print('Selection locked:', selection['selected'], flush=True)
        if phase == 'development':
            return selection
    selection_text = selection_path.read_bytes()
    selection = json.loads(selection_text)
    if (selection['archive_sha256'] != sha256 or selection['train_rows'] != train_rows or
            selection['calibration_rows'] != calibration_rows or selection['development_rows'] != development_rows or
            selection['max_rows'] != max_rows or selection['alphas'] != list(ALPHAS) or
            selection['context_columns'] != CONTEXT_COLUMNS):
        raise ValueError('Frozen selection does not match this run')
    data, info = replay(archive_path, max_rows, train_rows, calibration_rows, development_rows, chunksize)
    holdout = data['source_row'] > development_rows
    names = list(dict.fromkeys(['forest_all','forest_success','global_surprisal',selection['selected']]))
    rows, groups, daily, positives = [], [], [], []
    midpoint = development_rows + (max_rows-development_rows)//2
    days = pd.to_datetime(data['time']).floor('D')
    for name in names:
        threshold = selection['thresholds'][name]
        row = metrics(data, holdout, name, threshold)
        rate = selection['calibration_eligible_alert_rates'][name]
        row['calibration_success_alert_rate'] = rate if name != 'forest_all' else None
        row['calibration_all_alert_rate'] = rate if name == 'forest_all' else None
        rows.append(row)
        masks = {'early': holdout & (data['source_row'] <= midpoint),
                 'late': holdout & (data['source_row'] > midpoint),
                 'no_prior_success':holdout & (data['prior']==0),
                 'has_prior_success':holdout & (data['prior']>0),
                 'train_top_id':holdout & data['top'], 'other_ids':holdout & ~data['top']}
        for group, mask in masks.items():
            groups.append({'group':group, **metrics(data,mask,name,threshold)})
        alerts = predictions(data,name,threshold)
        table = pd.DataFrame({'day':days[holdout], 'alert':alerts[holdout],
                              'known':data['known'][holdout], 'success':data['success'][holdout]})
        counts = table.groupby('day').agg(events=('alert','size'),alerts=('alert','sum'),
                                          known_positives=('known','sum'),success_events=('success','sum')).reset_index()
        counts['model'] = name
        daily.append(counts)
        mask = holdout & data['known']
        positives.append(pd.DataFrame({'model':name, 'source_row':data['source_row'][mask],
                                       'prior_successes':data['prior'][mask], 'success':data['success'][mask],
                                       'score':(data['forest'] if name.startswith('forest_') else data[name])[mask],
                                       'alert':alerts[mask]}))
    pd.DataFrame(rows).to_csv(output/'holdout_metrics.csv',index=False)
    pd.DataFrame(groups).to_csv(output/'holdout_groups.csv',index=False)
    pd.concat(daily,ignore_index=True).to_csv(output/'holdout_daily.csv',index=False)
    pd.concat(positives,ignore_index=True).to_csv(output/'holdout_positive_details.csv',index=False)
    info.update(archive_sha256=sha256, archive_md5=md5, publisher_md5_verified=expected_md5 is not None,
                selection_sha256=hashlib.sha256(selection_text).hexdigest(), holdout_rows=int(holdout.sum()),
                holdout_known_positive_failures=int((holdout & data['known'] & ~data['success']).sum()),
                note='Same synthesized dataset; partial positive labels; no FPR or precision.')
    (output/'audit.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    return pd.DataFrame(rows), info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('development','holdout','both'),default='both')
    run(phase=parser.parse_args().phase)
