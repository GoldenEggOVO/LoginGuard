import hashlib
import json
from pathlib import Path

import pandas as pd

from .rba_context import replay, metrics, predictions
from .rba_validation import archive_hashes, PUBLISHER_MD5, alert_budget_cutoff
from .experiment_runs import experiment_run, resolve_run


def lock_budget(data, parent):
    eligible = (data['source_row'] <= parent['calibration_rows']) & data['success']
    budget = parent['calibration_eligible_alert_rates']['forest_success']
    thresholds = dict(parent['thresholds'])
    rates = {}
    for name, source in [('global_matched','global_surprisal'),('personal_matched',parent['selected'])]:
        scores = data[source][eligible]
        thresholds[name] = alert_budget_cutoff(scores,budget)
        rates[name] = float((scores >= thresholds[name]).mean())
    return {'target_budget':budget,'thresholds':thresholds,'rates':rates,'selected':parent['selected']}


def run(root=Path('.'), parent_path=None,
        archive_path=Path('data/public/rba/rba-dataset.zip')):
    if parent_path is None:
        parent_path = resolve_run(Path(root)/'reports/rba_context')/'selection.json'
    with experiment_run(Path(root)/'reports/rba_budget','budget') as output:
        return _run(output,parent_path,archive_path)


def _run(output, parent_path, archive_path):
    parent_bytes = Path(parent_path).read_bytes()
    parent = json.loads(parent_bytes)
    boundaries = tuple(parent.get(name) for name in
                       ('train_rows','calibration_rows','development_rows','max_rows'))
    # Calibration replay and the new holdout use the frozen parent row protocol.
    if any(type(value) is not int for value in boundaries) or boundaries != (1_200_000,2_000_000,4_000_000,8_000_000):
        raise ValueError('Unsupported parent row protocol; expected integer train/calibration/development/max rows '
                         'of 1200000/2000000/4000000/8000000')
    sha256,md5 = archive_hashes(archive_path)
    if md5 != PUBLISHER_MD5:
        raise ValueError('Archive MD5 does not match expected publisher checksum')
    if sha256 != parent['archive_sha256']:
        raise ValueError('Archive differs from the parent study')
    data,_ = replay(archive_path,4_000_000,parent['train_rows'],parent['calibration_rows'],
                    parent['development_rows'],100_000)
    lock = lock_budget(data,parent)
    lock.update(parent_selection_sha256=hashlib.sha256(parent_bytes).hexdigest(),
                archive_sha256=sha256, holdout_start_row=8_000_001,holdout_end_row=12_000_000,
                context_source_sha256=hashlib.sha256(Path(__file__).with_name('rba_context.py').read_bytes()).hexdigest())
    selection_path = output/'selection.json'
    selection_path.write_text(json.dumps(lock,indent=2),encoding='utf-8')
    del data
    print('Matched-budget cutoffs locked before new holdout:',lock['rates'],flush=True)
    data,info = replay(archive_path,12_000_000,parent['train_rows'],parent['calibration_rows'],
                       parent['development_rows'],100_000,extra_boundaries=(8_000_000,))
    data['global_matched'] = data['global_surprisal']
    data['personal_matched'] = data[parent['selected']]
    names = ['forest_all','forest_success','global_surprisal',parent['selected'],'global_matched','personal_matched']
    holdout = data['source_row'] > 8_000_000
    rows,groups,details = [],[],[]
    for name in names:
        threshold = lock['thresholds'][name]
        rows.append(metrics(data,holdout,name,threshold))
        masks = {'early':holdout & (data['source_row']<=10_000_000),
                 'late':holdout & (data['source_row']>10_000_000),
                 'no_prior_success':holdout & (data['prior']==0),
                 'has_prior_success':holdout & (data['prior']>0),
                 'train_top_id':holdout & data['top'],'other_ids':holdout & ~data['top']}
        for group,mask in masks.items():
            groups.append({'group':group,**metrics(data,mask,name,threshold)})
        mask = holdout & data['known']
        details.append(pd.DataFrame({'model':name,'source_row':data['source_row'][mask],
                                     'prior_successes':data['prior'][mask],
                                     'alert':predictions(data,name,threshold)[mask]}))
    result = pd.DataFrame(rows)
    result.to_csv(output/'holdout_metrics.csv',index=False)
    pd.DataFrame(groups).to_csv(output/'holdout_groups.csv',index=False)
    pd.concat(details,ignore_index=True).to_csv(output/'holdout_positive_details.csv',index=False)
    info.update(archive_sha256=sha256,archive_md5=md5,publisher_md5_verified=True,
                selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                holdout_rows=int(holdout.sum()),
                first_holdout_timestamp=str(pd.Timestamp(data['time'][holdout][0])),
                holdout_known_positive_failures=int((holdout & data['known'] & ~data['success']).sum()))
    (output/'audit.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(result.to_string(index=False),flush=True)
    return result,info


if __name__ == '__main__':
    run()
