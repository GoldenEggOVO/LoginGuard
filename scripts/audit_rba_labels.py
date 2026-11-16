import json
import hashlib
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from src.rba_validation import archive_hashes
from src.experiment_runs import experiment_run, resolve_run


def run():
    base = Path('reports/rba_budget')
    source = resolve_run(base)
    with experiment_run(base/'label_audits','analysis') as output:
        (output/'inputs.json').write_text(json.dumps({name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                for name in ['audit.json','holdout_positive_details.csv']},indent=2),encoding='utf-8')
        _run(source,output)


def _run(source,output):
    archive_path = Path('data/public/rba/rba-dataset.zip')
    audit = json.loads((source/'audit.json').read_text(encoding='utf-8'))
    sha256,_ = archive_hashes(archive_path)
    if sha256 != audit['archive_sha256']:
        raise ValueError('Archive differs from the evaluated dataset')
    offset,known_success,failures = 0,0,[]
    with ZipFile(archive_path) as archive, archive.open('rba-dataset.csv') as stream:
        for chunk in pd.read_csv(stream,usecols=['Login Successful','Is Account Takeover'],
                                  dtype=str,keep_default_na=False,chunksize=100_000,nrows=12_000_000):
            known = chunk['Is Account Takeover'].str.strip().str.lower().eq('true').to_numpy()
            success = chunk['Login Successful'].str.strip().str.lower().eq('true').to_numpy()
            rows = offset + np.arange(1,len(chunk)+1)
            selected = rows > 8_000_000
            known_success += int((selected & known & success).sum())
            failures.extend(rows[selected & known & ~success].tolist())
            offset += len(chunk)
    details = pd.read_csv(source/'holdout_positive_details.csv')
    success_counts = details[~details.source_row.isin(failures)].groupby('model').alert.agg(['size','sum'])
    failure_counts = details[details.source_row.isin(failures)].groupby('model').alert.sum()
    if not (success_counts['size'] == known_success).all():
        raise ValueError('Label audit does not align with evaluated records')
    result = {'holdout_known_successful_records':known_success,'holdout_known_failed_records':len(failures),
              'known_failed_source_rows':failures,
              'successful_known_alerts':{k:int(v) for k,v in success_counts['sum'].items()},
              'failed_known_alerts':{k:int(v) for k,v in failure_counts.items()},
              'note':'Preserved labels; no rows relabelled or excluded. Success-gated evaluation also reports the all-labelled denominator.'}
    (output/'label_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    run()
