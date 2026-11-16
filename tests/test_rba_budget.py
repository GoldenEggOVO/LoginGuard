import json
from pathlib import Path

import numpy as np
import pytest


def test_matching_budget_uses_only_calibration_success_scores():
    from src.rba_budget import lock_budget
    data = {'source_row':np.arange(101,401),'success':np.ones(300,dtype=bool),
            'global_surprisal':np.arange(300,dtype=float),
            'surprisal_a10':np.arange(300,dtype=float),'known':np.zeros(300,dtype=bool)}
    parent = {'selected':'surprisal_a10','calibration_rows':200,
              'calibration_eligible_alert_rates':{'forest_success':.015},
              'thresholds':{'forest_all':.8,'forest_success':.7,'global_surprisal':80.,'surprisal_a10':75.}}
    one = lock_budget(data,parent)
    data['surprisal_a10'][100:] = -1000
    data['global_surprisal'][100:] = -1000
    data['known'][:] = True
    two = lock_budget(data,parent)
    assert one == two
    assert one['rates']['personal_matched'] <= .015
    assert one['thresholds']['personal_matched'] > parent['thresholds']['surprisal_a10']


@pytest.mark.parametrize('changes',[
    {'development_rows':10_000_000},
    {'development_rows':4_000_001},
    {'train_rows':0},
    {'calibration_rows':1_200_000},
    {'development_rows':2_000_000},
    {'max_rows':4_000_000},
    {'train_rows':1_000_000},
    {'calibration_rows':1_800_000},
    {'development_rows':3_000_000},
    {'max_rows':7_000_000},
    {'max_rows':12_000_000},
    *[{field:value} for field,expected in [
        ('train_rows',1_200_000),('calibration_rows',2_000_000),
        ('development_rows',4_000_000),('max_rows',8_000_000)]
      for value in [True,float(expected),str(expected),None]],
])
def test_unsupported_parent_protocol_is_rejected_before_archive_or_output(tmp_path,monkeypatch,changes):
    from src import rba_budget
    parent = {'train_rows':1_200_000,'calibration_rows':2_000_000,
              'development_rows':4_000_000,'max_rows':8_000_000} | changes
    parent_path = tmp_path/'parent.json'
    parent_path.write_text(json.dumps(parent),encoding='utf-8')
    output = tmp_path/'output'
    output.mkdir()
    (output/'selection.json').write_bytes(b'previous selection')
    def forbidden(*args,**kwargs):
        pytest.fail('Unsupported parent protocol reached archive hashing or replay')
    monkeypatch.setattr(rba_budget,'archive_hashes',forbidden)
    monkeypatch.setattr(rba_budget,'replay',forbidden)
    with pytest.raises(ValueError,match='parent.*protocol'):
        rba_budget._run(output,parent_path,tmp_path/'unused.zip')
    assert {path.name:path.read_bytes() for path in output.iterdir()} == {
        'selection.json':b'previous selection'}


def test_archived_parent_protocol_reaches_fixed_calibration_replay(tmp_path,monkeypatch):
    from src import rba_budget
    parent_path = Path('reports/rba_context/selection.json')
    parent = json.loads(parent_path.read_bytes())
    monkeypatch.setattr(rba_budget,'archive_hashes',
                        lambda _: (parent['archive_sha256'],rba_budget.PUBLISHER_MD5))
    def replay(archive,max_rows,train_rows,calibration_rows,development_rows,chunksize):
        assert (max_rows,train_rows,calibration_rows,development_rows,chunksize) == (
            4_000_000,1_200_000,2_000_000,4_000_000,100_000)
        raise RuntimeError('archived parent accepted')
    monkeypatch.setattr(rba_budget,'replay',replay)
    with pytest.raises(RuntimeError,match='archived parent accepted'):
        rba_budget._run(tmp_path,parent_path,tmp_path/'unused.zip')
