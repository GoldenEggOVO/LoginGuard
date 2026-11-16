import numpy as np
import pandas as pd
import pytest

from src.rba_context import ContextRisk


def test_tied_scores_use_exact_label_independent_calibration_count():
    from src.rba_fair import calibrate_exact, predict_exact, tie_keys
    rows = np.arange(1, 101)
    scores = np.ones(100)
    lock = calibrate_exact(scores, rows, .02)
    assert predict_exact(scores, rows, lock).sum() == 2
    reverse = np.arange(99, -1, -1)
    assert calibrate_exact(scores[reverse], rows[reverse], .02) == lock
    assert len(np.unique(tie_keys(rows))) == len(rows)
    assert not predict_exact(scores, rows, calibrate_exact(scores, rows, .005)).any()


def test_frozen_tie_decisions_do_not_use_future_batch_composition():
    from src.rba_fair import calibrate_exact, predict_exact
    lock = calibrate_exact(np.ones(100), np.arange(1, 101), .02)
    whole = predict_exact(np.ones(100), np.arange(101, 201), lock)
    prefix = predict_exact(np.ones(30), np.arange(101, 131), lock)
    np.testing.assert_array_equal(whole[:30], prefix)


@pytest.mark.parametrize('rows', [np.array([1.2, 1.4]), np.array([True, False]), np.array([0, 1]), np.array([-1, 1])])
def test_exact_budget_rejects_nonordinal_row_keys(rows):
    from src.rba_fair import calibrate_exact
    with pytest.raises(ValueError, match='row ordinals'):
        calibrate_exact(np.ones(2), rows, .5)


def test_per_field_vectors_reproduce_the_existing_scalar_scores_and_state():
    plain, expanded = ContextRisk(), ContextRisk()
    observations = [(['ip1', 'asn1', 'US', 'os1', 'browser1', 'desktop'], True),
                    (['ip2', 'unknown', 'US', '', 'browser2', 'desktop'], False),
                    (['ip2', 'asn2', 'US', 'os1', 'browser2', 'desktop'], True)]
    for values, success in observations:
        expected, old_prior = plain.score_update('user', values, success)
        actual, prior = expanded.score_update('user', values, success, include_fields=True)
        available = actual.pop('available_fields')
        global_fields, personal_fields = actual.pop('global_fields'), actual.pop('personal_fields')
        assert prior == old_prior
        assert actual == expected
        np.testing.assert_allclose(sum(global_fields) / sum(available), expected['global_surprisal'])
        np.testing.assert_allclose(sum(personal_fields) / sum(available), expected['surprisal_a10'])
    assert expanded.global_counts == plain.global_counts
    assert expanded.user_counts == plain.user_counts


def make_archive(tmp_path, equal_boundary=False, invalid_boundary=False):
    from zipfile import ZipFile
    times = pd.date_range('2020-01-01', periods=12, freq='s').astype(str).tolist()
    if equal_boundary:
        times[3] = times[2]
    raw = pd.DataFrame({'Login Timestamp': times, 'User ID': ['u1', 'u2'] * 6,
                        'IP Address': ['192.0.2.1', '192.0.2.2'] * 6,
                        'Login Successful': ['True', 'True', 'False'] * 4,
                        'Is Account Takeover': ['False'] * 12,
                        'ASN': ['1'] * 12, 'Country': ['US'] * 12,
                        'OS Name and Version': ['os'] * 12, 'Browser Name and Version': ['browser'] * 12,
                        'Device Type': ['desktop'] * 12})
    if invalid_boundary:
        raw.loc[3, 'IP Address'] = 'invalid'
    path = tmp_path / 'fixture.zip'
    with ZipFile(path, 'w') as archive:
        archive.writestr('rba-dataset.csv', raw.to_csv(index=False))
    return path


def test_context_vectors_and_history_are_chunk_invariant(tmp_path):
    from src.rba_fair import batches
    archive = make_archive(tmp_path)
    def replay(size):
        pieces = list(batches(archive, 12, (3, 6, 9), chunksize=size))
        return {name: np.concatenate([part[name] for part in pieces])
                for name in ('base', 'global', 'personal', 'prior')}
    small, whole = replay(2), replay(12)
    for name in small:
        np.testing.assert_array_equal(small[name], whole[name])


def test_new_replay_rejects_equal_timestamp_split(tmp_path):
    from src.rba_fair import batches
    with pytest.raises(ValueError, match='equal timestamps'):
        list(batches(make_archive(tmp_path, equal_boundary=True), 12, (3, 6, 9), chunksize=2))


def test_invalid_boundary_record_cannot_hide_raw_timestamp_tie(tmp_path):
    from src.rba_fair import batches
    with pytest.raises(ValueError, match='equal timestamps'):
        list(batches(make_archive(tmp_path, equal_boundary=True, invalid_boundary=True),
                     12, (3, 6, 9), chunksize=2))


def test_score_methods_accepts_failure_only_batches():
    from src.rba_fair import score_methods, FORESTS
    class Forest:
        def score_samples(self, rows):
            assert len(rows) > 0
            return np.ones(len(rows))
    result = score_methods({'base': np.empty((0, 5)), 'global': np.empty((0, 12)),
                            'personal': np.empty((0, 12))}, dict.fromkeys(FORESTS, Forest()))
    assert all(len(values) == 0 for values in result.values())


def test_public_selection_is_written_before_new_range_and_ignores_calibration_labels(tmp_path, monkeypatch):
    import json
    from src import rba_fair
    from src.experiment_runs import resolve_run
    monkeypatch.setattr(rba_fair, 'archive_hashes', lambda _: ('fixture', rba_fair.PUBLISHER_MD5))
    flip = [False]
    def batch(start, end, count):
        values = np.arange(count, dtype=float)
        vectors = np.column_stack([np.repeat(values[:, None], 6, axis=1), np.ones((count, 6))])
        rows = np.arange(start + 1, start + count + 1)
        return {'start': start, 'end': end, 'events': pd.DataFrame({'source_row': rows}),
                'labels': pd.DataFrame({'label_state': ['known_positive' if flip[0] else 'unlabelled_explicit_false'] * count}),
                'eligible': np.ones(count, dtype=bool), 'base': np.repeat(values[:, None], 5, axis=1),
                'global': vectors, 'personal': vectors, 'prior': np.zeros(count, dtype=int),
                'excluded': 0, 'excluded_known': 0}
    def stream(*args):
        yield batch(0, rba_fair.TRAIN_END, 12)
        yield batch(rba_fair.TRAIN_END, rba_fair.CALIBRATION_END, 100)
        paths = [path for path in (tmp_path / 'reports/rba_fair/runs').glob('*/selection.json')
                 if not (path.parent / 'COMPLETE.json').exists()]
        assert len(paths) == 1
        lock = json.loads(paths[0].read_bytes())
        assert lock['evaluation_start'] == 12_000_000
        for cutoffs in lock['cutoffs'].values():
            assert [entry['allowed'] for entry in cutoffs] == [0, 1, 2]
        flip[0] = False
        yield batch(rba_fair.EVALUATION_START, rba_fair.EVALUATION_END, 20)
    monkeypatch.setattr(rba_fair, 'batches', stream)
    rba_fair.run('unused.zip', root=tmp_path)
    base = tmp_path / 'reports/rba_fair'
    first = resolve_run(base)
    original = json.loads((first / 'selection.json').read_bytes())
    flip[0] = True
    rba_fair.run('unused.zip', root=tmp_path)
    repeated = json.loads((resolve_run(base) / 'selection.json').read_bytes())
    assert original == repeated
    assert (first / 'metrics.csv').is_file()
