"""Frozen, success-only context controls on a new chronological RBA range."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

from .experiment_runs import experiment_run
from .rba_context import CONTEXT_COLUMNS, ContextRisk
from .rba_validation import RAW_COLUMNS, PUBLISHER_MD5, adapt_rows, archive_hashes, chronological_features

TRAIN_END, CALIBRATION_END, EVALUATION_START, EVALUATION_END = 1_200_000, 2_000_000, 12_000_000, 16_000_000
BUDGETS = (.005, .01, .02)
TIE_SEED = 20261114
FORESTS = ('count_forest_successfit', 'global_field_forest', 'personal_field_forest')
METHODS = FORESTS + ('global_mean', 'personal_mean')


def tie_keys(source_rows):
    """SplitMix64 is a bijection: unique row ordinals have unique tie keys."""
    values = np.asarray(source_rows, dtype=np.uint64) ^ np.uint64(TIE_SEED)
    with np.errstate(over='ignore'):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> 27)) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> 31)


def calibrate_exact(scores, source_rows, budget):
    scores, rows = np.asarray(scores), np.asarray(source_rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or np.any(rows <= 0):
        raise ValueError('Calibration row ordinals must be positive integers')
    if (scores.ndim != 1 or rows.ndim != 1 or not len(scores) or len(scores) != len(rows)
            or not np.isfinite(scores).all() or len(np.unique(rows)) != len(rows) or not 0 <= budget < 1):
        raise ValueError('Calibration requires finite scores, unique rows, and a valid budget')
    allowed = int(np.floor(budget * len(scores)))
    lock = {'budget': budget, 'eligible': len(scores), 'allowed': allowed, 'score': None, 'tie_key': None}
    if allowed:
        keys = tie_keys(rows)
        boundary = np.lexsort((keys, -scores))[allowed - 1]
        lock.update(score=float(scores[boundary]), tie_key=int(keys[boundary]))
    return lock


def predict_exact(scores, source_rows, lock):
    scores, rows = np.asarray(scores), np.asarray(source_rows)
    if not lock['allowed']:
        return np.zeros(len(scores), dtype=bool)
    tied = scores == lock['score']
    selected = scores > lock['score']
    selected[tied] = tie_keys(rows[tied]) <= np.uint64(lock['tie_key'])
    return selected


def batches(archive_path, max_rows, boundaries, chunksize=100_000):
    """Carry histories across chunks; reject a split that cuts equal timestamps."""
    context, window = ContextRisk(), {}
    offset, previous, partition = 0, None, 0
    previous_raw_timestamp = None
    excluded, excluded_known = 0, 0
    with ZipFile(archive_path) as archive, archive.open('rba-dataset.csv') as stream:
        reader = pd.read_csv(stream, usecols=list(dict.fromkeys(RAW_COLUMNS + CONTEXT_COLUMNS)),
                             dtype=str, keep_default_na=False, chunksize=chunksize, nrows=max_rows)
        for chunk in reader:
            cuts = [0] + [point - offset for point in boundaries if offset < point < offset + len(chunk)] + [len(chunk)]
            for left, right in zip(cuts, cuts[1:]):
                start, end = offset + left, offset + right
                raw = chunk.iloc[left:right].reset_index(drop=True)
                # Raw ordinals define the protocol, including records later excluded by adaptation.
                first_raw_timestamp = pd.to_datetime(raw['Login Timestamp'].iloc[0], errors='coerce')
                if start in boundaries and pd.notna(first_raw_timestamp) and first_raw_timestamp == previous_raw_timestamp:
                    raise ValueError('A raw split boundary cuts equal timestamps')
                previous_raw_timestamp = pd.to_datetime(raw['Login Timestamp'].iloc[-1], errors='coerce')
                events, labels, audit = adapt_rows(raw)
                excluded += audit['excluded_rows']
                excluded_known += audit['excluded_known_positive']
                values = raw[CONTEXT_COLUMNS].iloc[events.source_row.to_numpy() - 1].to_numpy()
                events['source_row'] += start
                if len(events):
                    current_partition = sum(start >= point for point in boundaries)
                    if previous is not None and (events.timestamp.iloc[0] < previous or
                            (current_partition > partition and events.timestamp.iloc[0] == previous)):
                        raise ValueError('Input is not chronological or a split cuts equal timestamps')
                    partition, previous = current_partition, events.timestamp.iloc[-1]
                base = chronological_features(events, window)
                eligible = events.success.to_numpy(dtype=bool)
                active = start < boundaries[1] or start >= boundaries[2]
                global_fields, personal_fields, priors = [], [], []
                for user, success, fields in zip(events.user_id, eligible, values):
                    priors.append(context.successes[user])
                    if success:
                        scores, _ = context.score_update(user, fields, True, include_fields=active)
                        if active:
                            global_fields.append(scores['global_fields'] + scores['available_fields'])
                            personal_fields.append(scores['personal_fields'] + scores['available_fields'])
                yield {'start': start, 'end': end, 'events': events, 'labels': labels, 'eligible': eligible,
                       'base': base.to_numpy()[eligible], 'global': np.asarray(global_fields, dtype=float).reshape(-1, 12),
                       'personal': np.asarray(personal_fields, dtype=float).reshape(-1, 12),
                       'prior': np.asarray(priors), 'excluded': excluded, 'excluded_known': excluded_known}
            offset += len(chunk)
            if offset % 1_000_000 == 0:
                print(f'Replayed {offset:,} rows', flush=True)
    if offset < max_rows:
        raise ValueError('Archive has insufficient rows for the frozen protocol')


def score_methods(batch, models):
    inputs = dict(zip(FORESTS, (batch['base'], batch['global'], batch['personal'])))
    result = {name: -model.score_samples(inputs[name]) if len(inputs[name]) else np.empty(0)
              for name, model in models.items()}
    for name, vector in (('global_mean', batch['global']), ('personal_mean', batch['personal'])):
        available = vector[:, 6:].sum(axis=1)
        result[name] = np.divide(vector[:, :6].sum(axis=1), available,
                                 out=np.zeros(len(vector)), where=available > 0)
    return result


def peak_memory_bytes():
    if platform.system() == 'Windows':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                    'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return counters.PeakWorkingSetSize
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == 'Darwin' else value * 1024


def run(archive_path=Path('data/public/rba/rba-dataset.zip'), root=Path('.')):
    started = perf_counter()
    sha256, md5 = archive_hashes(archive_path)
    if md5 != PUBLISHER_MD5:
        raise ValueError('Archive MD5 does not match expected publisher checksum')
    source = Path(__file__).resolve().parent
    with experiment_run(Path(root) / 'reports/rba_fair', 'holdout') as output:
        train = {name: [] for name in FORESTS}
        fitted_records = {}
        first_evaluation_time = last_evaluation_time = None
        calibration = {name: [] for name in METHODS}
        calibration_rows, models, locks, outcomes = [], {}, None, []
        selection_bytes = None
        for batch in batches(archive_path, EVALUATION_END, (TRAIN_END, CALIBRATION_END, EVALUATION_START)):
            start, end = batch['start'], batch['end']
            inputs = dict(zip(FORESTS, (batch['base'], batch['global'], batch['personal'])))
            if start < TRAIN_END:
                for name in FORESTS:
                    train[name].append(inputs[name])
                if end == TRAIN_END:
                    for name in FORESTS:
                        model = IsolationForest(n_estimators=200, random_state=42, contamination='auto', n_jobs=1)
                        matrix = np.concatenate(train[name])
                        fitted_records[name] = len(matrix)
                        model.fit(matrix)
                        joblib.dump(model, output / f'{name}.joblib')
                        models[name] = model
                    train.clear()
            elif start < CALIBRATION_END:
                scores = score_methods(batch, models)
                calibration_rows.append(batch['events'].source_row.to_numpy()[batch['eligible']])
                for name in METHODS:
                    calibration[name].append(scores[name])
                if end == CALIBRATION_END:
                    rows = np.concatenate(calibration_rows)
                    locks = {name: [calibrate_exact(np.concatenate(calibration[name]), rows, budget)
                                    for budget in BUDGETS] for name in METHODS}
                    selection = {'archive_sha256': sha256, 'archive_md5': md5, 'train_end': TRAIN_END,
                                 'calibration_end': CALIBRATION_END, 'evaluation_start': EVALUATION_START,
                                 'evaluation_end': EVALUATION_END, 'alpha': 10, 'tie_seed': TIE_SEED,
                                 'primary_budget': .02, 'cutoffs': locks,
                                 'context_fields': CONTEXT_COLUMNS, 'context_vector': 'six surprisals then six availability indicators',
                                 'forest_fit_population': 'successful training records only',
                                 'forest_fit_records': fitted_records,
                                 'protocol_sha256': hashlib.sha256((source.parent / 'docs/fair_calibration_protocol.md').read_bytes()).hexdigest(),
                                 'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                                   for name in ('rba_fair.py', 'rba_context.py', 'rba_validation.py', 'experiment_runs.py')},
                                 'model_sha256': {name: hashlib.sha256((output / f'{name}.joblib').read_bytes()).hexdigest()
                                                  for name in FORESTS}}
                    selection_bytes = json.dumps(selection, indent=2).encode()
                    (output / 'selection.json').write_bytes(selection_bytes)
                    calibration.clear()
                    calibration_rows.clear()
                    print('Selection frozen at row 2,000,000; evaluation begins after row 12,000,000', flush=True)
            elif start >= EVALUATION_START:
                if locks is None:
                    raise ValueError('Calibration must be frozen before evaluation')
                scores = score_methods(batch, models)
                if len(batch['events']) and 'timestamp' in batch['events']:
                    if first_evaluation_time is None:
                        first_evaluation_time = str(batch['events'].timestamp.iloc[0])
                    last_evaluation_time = str(batch['events'].timestamp.iloc[-1])
                eligible = batch['eligible']
                rows = batch['events'].source_row.to_numpy()[eligible]
                known = batch['labels'].label_state.eq('known_positive').to_numpy()
                for name in METHODS:
                    for lock in locks[name]:
                        alerts = np.zeros(len(eligible), dtype=bool)
                        alerts[eligible] = predict_exact(scores[name], rows, lock)
                        for group, mask in (('all', np.ones(len(eligible), dtype=bool)),
                                            ('no_prior_success', batch['prior'] == 0),
                                            ('has_prior_success', batch['prior'] > 0)):
                            outcomes.append({'method': name, 'calibration_budget': lock['budget'], 'group': group,
                                             'events': int(mask.sum()), 'success_events': int((mask & eligible).sum()),
                                             'known_positives': int((mask & known).sum()),
                                             'known_positive_failures': int((mask & known & ~eligible).sum()),
                                             'known_positives_alerted': int((mask & known & alerts).sum()),
                                             'unlabelled_events': int((mask & ~known).sum()),
                                             'unlabelled_alerts': int((mask & ~known & alerts).sum()),
                                             'unlabelled_success_events': int((mask & ~known & eligible).sum())})
        if selection_bytes is None:
            raise ValueError('No calibration selection was produced')
        assert (output / 'selection.json').read_bytes() == selection_bytes
        columns = ['method', 'calibration_budget', 'group']
        summary = pd.DataFrame(outcomes).groupby(columns, sort=False).sum(numeric_only=True).reset_index()
        summary['known_positive_discovery'] = summary.known_positives_alerted / summary.known_positives.replace(0, np.nan)
        summary['unlabelled_alert_rate'] = summary.unlabelled_alerts / summary.unlabelled_events.replace(0, np.nan)
        summary.to_csv(output / 'metrics.csv', index=False)
        audit = {'archive_sha256': sha256, 'source_rows': EVALUATION_END,
                 'excluded_rows': batch['excluded'], 'excluded_known_positive': batch['excluded_known'],
                 'selection_sha256': hashlib.sha256(selection_bytes).hexdigest(),
                 'elapsed_seconds': perf_counter() - started, 'peak_working_set_bytes': peak_memory_bytes(),
                 'platform': platform.system(), 'python': platform.python_version(),
                 'logical_cpus': os.cpu_count(), 'processor': platform.processor(), 'chunksize': 100_000,
                 'first_evaluation_timestamp': first_evaluation_time, 'last_evaluation_timestamp': last_evaluation_time,
                 'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'scikit-learn': sklearn.__version__}}
        (output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        print(summary[summary.group.eq('all')].to_string(index=False), flush=True)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('data/public/rba/rba-dataset.zip'))
    parser.add_argument('--root', type=Path, default=Path('.'))
    options = parser.parse_args()
    run(options.archive, options.root)
