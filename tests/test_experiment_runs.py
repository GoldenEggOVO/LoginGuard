import json
from pathlib import Path

import numpy as np
import pytest

from src import rba_budget, rba_context


def parent_selection():
    return {'selected':'surprisal_a10','archive_sha256':'archive',
            'train_rows':1_200_000,'calibration_rows':2_000_000,'development_rows':4_000_000,'max_rows':8_000_000,
            'thresholds':{'forest_all':.6,'forest_success':.4,'global_surprisal':90.,'surprisal_a10':85.},
            'calibration_eligible_alert_rates':{'forest_success':.02}}


def calibration_data():
    return {'source_row':np.arange(1_200_001,1_200_101),'success':np.ones(100,dtype=bool),
            'global_surprisal':np.arange(100,dtype=float),'surprisal_a10':np.arange(100,dtype=float)}


def old_results(root, name):
    output = root/'reports'/name
    output.mkdir(parents=True)
    (output/'selection.json').write_text('{"old":true}',encoding='utf-8')
    (output/'holdout_metrics.csv').write_text('old,metrics\n1,2\n',encoding='utf-8')
    return output, {f.name:f.read_bytes() for f in output.iterdir()}


def test_failed_budget_replay_preserves_previous_success(tmp_path, monkeypatch):
    output,old = old_results(tmp_path,'rba_budget')
    parent = tmp_path/'parent.json'
    parent.write_text(json.dumps(parent_selection()),encoding='utf-8')
    monkeypatch.setattr(rba_budget,'archive_hashes',lambda _:('archive',rba_budget.PUBLISHER_MD5))
    calls = []
    def replay(*args,**kwargs):
        calls.append(args[1])
        if len(calls)==1:
            return calibration_data(),{}
        raise RuntimeError('interrupted holdout')
    monkeypatch.setattr(rba_budget,'replay',replay)
    with pytest.raises(RuntimeError,match='interrupted'):
        rba_budget.run(root=tmp_path,parent_path=parent)
    assert calls == [4_000_000,12_000_000]
    assert {name:(output/name).read_bytes() for name in old} == old
    assert not (output/'latest_complete.json').exists()


def test_failed_context_holdout_preserves_previous_success(tmp_path, monkeypatch):
    output,old = old_results(tmp_path,'rba_context')
    selection = parent_selection() | {'max_rows':8_000_000,'alphas':list(rba_context.ALPHAS),
                                    'context_columns':rba_context.CONTEXT_COLUMNS,'development_metrics':[]}
    monkeypatch.setattr(rba_context,'archive_hashes',lambda _:('archive',rba_context.PUBLISHER_MD5))
    monkeypatch.setattr(rba_context,'develop',lambda *args:dict(selection))
    calls=[]
    def replay(*args,**kwargs):
        calls.append(args[1])
        if len(calls)==1:
            return {},{}
        raise RuntimeError('interrupted holdout')
    monkeypatch.setattr(rba_context,'replay',replay)
    with pytest.raises(RuntimeError,match='interrupted'):
        rba_context.run(root=tmp_path)
    assert {name:(output/name).read_bytes() for name in old} == old
    assert not (output/'latest_complete.json').exists()


def test_manifest_binds_complete_files_and_failure_never_advances_pointer(tmp_path):
    from src.experiment_runs import experiment_run, resolve_run
    with experiment_run(tmp_path,'holdout') as output:
        (output/'selection.json').write_text('first')
        (output/'metrics.csv').write_text('complete')
    first = resolve_run(tmp_path)
    pointer = (tmp_path/'latest_complete.json').read_bytes()
    manifest = json.loads((first/'COMPLETE.json').read_text())
    assert set(manifest['files']) == {'selection.json','metrics.csv'}
    with pytest.raises(RuntimeError):
        with experiment_run(tmp_path,'holdout') as partial:
            (partial/'selection.json').write_text('second')
            raise RuntimeError('stop')
    assert resolve_run(tmp_path)==first
    assert (tmp_path/'latest_complete.json').read_bytes()==pointer
    assert not (partial/'COMPLETE.json').exists()
    assert (partial/'FAILED.json').exists()
    (first/'metrics.csv').write_text('edited')
    with pytest.raises(ValueError,match='checksum'):
        resolve_run(tmp_path)


def test_development_is_not_a_completed_holdout_and_bad_pointer_never_falls_back(tmp_path):
    from src.experiment_runs import experiment_run, resolve_run
    (tmp_path/'selection.json').write_text('legacy')
    assert resolve_run(tmp_path)==tmp_path
    with experiment_run(tmp_path,'development') as output:
        (output/'selection.json').write_text('new development')
    assert resolve_run(tmp_path)==tmp_path
    assert resolve_run(tmp_path,phase='development')==output
    (tmp_path/'latest_complete.json').write_text('{"run_id":"../outside"}')
    with pytest.raises(ValueError):
        resolve_run(tmp_path)


def test_normal_budget_output_is_complete_and_statistics_are_unchanged(tmp_path,monkeypatch):
    from io import StringIO
    import pandas as pd
    from src.experiment_runs import resolve_run
    output,old = old_results(tmp_path,'rba_budget')
    parent = tmp_path/'parent.json'
    parent.write_text(json.dumps(parent_selection()))
    data = {'source_row':np.array([8_000_001,8_000_002,8_000_003]),
            'time':pd.date_range('2020-01-01',periods=3,freq='1s').as_unit('ns').asi8,
            'success':np.array([True,False,True]),'known':np.array([True,False,True]),
            'prior':np.array([0,1,4]),'top':np.array([False,True,False]),
            'forest':np.array([.9,.1,.5]),'global_surprisal':np.array([110.,0.,120.]),
            'surprisal_a10':np.array([105.,0.,115.])}
    monkeypatch.setattr(rba_budget,'archive_hashes',lambda _:('archive',rba_budget.PUBLISHER_MD5))
    monkeypatch.setattr(rba_budget,'replay',lambda _,count,*a,**kw:(calibration_data() if count==4_000_000 else dict(data),{}))
    result,_ = rba_budget.run(root=tmp_path,parent_path=parent)
    complete = resolve_run(output)
    assert complete != output
    assert {name:(output/name).read_bytes() for name in old}==old
    stored = pd.read_csv(complete/'holdout_metrics.csv')
    pd.testing.assert_frame_equal(stored,pd.read_csv(StringIO(result.to_csv(index=False))))
    counts = result.set_index('model').known_positives_alerted.to_dict()
    assert counts == {'forest_all':1,'forest_success':2,'global_surprisal':2,
                      'surprisal_a10':2,'global_matched':2,'personal_matched':2}
    manifest = json.loads((complete/'COMPLETE.json').read_text())
    assert set(manifest['files'])=={'selection.json','holdout_metrics.csv','holdout_groups.csv',
                                   'holdout_positive_details.csv','audit.json'}


def test_separate_development_then_failed_holdout_keeps_old_results(tmp_path,monkeypatch):
    from src.experiment_runs import resolve_run
    output,old = old_results(tmp_path,'rba_context')
    selection = parent_selection() | {'max_rows':8_000_000,'alphas':list(rba_context.ALPHAS),
                                    'context_columns':rba_context.CONTEXT_COLUMNS,'development_metrics':[]}
    monkeypatch.setattr(rba_context,'archive_hashes',lambda _:('archive',rba_context.PUBLISHER_MD5))
    monkeypatch.setattr(rba_context,'develop',lambda *args:dict(selection))
    monkeypatch.setattr(rba_context,'replay',lambda *a,**kw:({},{}))
    rba_context.run(root=tmp_path,phase='development')
    development = resolve_run(output,phase='development')
    assert development!=output and resolve_run(output)==output
    def stop(*args,**kwargs):
        raise RuntimeError('holdout failed')
    monkeypatch.setattr(rba_context,'replay',stop)
    with pytest.raises(RuntimeError,match='holdout failed'):
        rba_context.run(root=tmp_path,phase='holdout')
    assert resolve_run(output)==output
    assert {name:(output/name).read_bytes() for name in old}==old


def test_plot_legacy_fallback_preserves_historical_image(tmp_path,monkeypatch):
    from scripts import plot_rba_context
    from src.experiment_runs import resolve_run
    source = Path('reports/rba_context')
    base = tmp_path/'reports/rba_context'
    base.mkdir(parents=True)
    for name in ['holdout_metrics.csv','holdout_groups.csv']:
        (base/name).write_bytes((source/name).read_bytes())
    (base/'holdout_comparison.png').write_bytes(b'historical image')
    monkeypatch.chdir(tmp_path)
    plot_rba_context.run()
    assert (base/'holdout_comparison.png').read_bytes()==b'historical image'
    assert (resolve_run(base/'plots')/'holdout_comparison.png').exists()


def test_label_audit_legacy_fallback_preserves_historical_audit(tmp_path,monkeypatch):
    from zipfile import ZipFile
    from scripts import audit_rba_labels
    from src.experiment_runs import resolve_run
    base = tmp_path/'reports/rba_budget'
    base.mkdir(parents=True)
    (base/'audit.json').write_text('{"archive_sha256":"sample"}')
    (base/'holdout_positive_details.csv').write_text('model,source_row,alert\n')
    (base/'label_audit.json').write_text('historical audit')
    archive = tmp_path/'data/public/rba/rba-dataset.zip'
    archive.parent.mkdir(parents=True)
    with ZipFile(archive,'w') as z:
        z.writestr('rba-dataset.csv','Login Successful,Is Account Takeover\nTrue,False\n')
    monkeypatch.setattr(audit_rba_labels,'archive_hashes',lambda _:('sample','md5'))
    monkeypatch.chdir(tmp_path)
    audit_rba_labels.run()
    assert (base/'label_audit.json').read_text()=='historical audit'
    assert (resolve_run(base/'label_audits')/'label_audit.json').exists()
