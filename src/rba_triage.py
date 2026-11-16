import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment_runs import experiment_run, resolve_run
from .rba_context import replay, predictions
from .rba_validation import archive_hashes, PUBLISHER_MD5
from .triage import group_alerts

BUDGETS = (10,50,100,None)
METHODS = ('forest_all','forest_success','global_matched','personal_matched')
ALERT_COLUMNS = ['event_id','timestamp','user_id','ip_address','rule_names','model_alert','score']


def _record_counts(truth, alerted, reviewed, mask):
    known = truth.known.to_numpy(dtype=bool)
    success = truth.success.to_numpy(dtype=bool)
    n_known = int((mask & known).sum())
    unknown_success = mask & ~known & success
    return {'events':int(mask.sum()),'success_events':int((mask & success).sum()),
            'alert_events':int((mask & alerted).sum()),'reviewed_alert_events':int((mask & reviewed).sum()),
            'known_labels':n_known,'known_alerted':int((mask & known & alerted).sum()),
            'known_reviewed':int((mask & known & reviewed).sum()),
            'known_record_coverage':float((mask & known & reviewed).sum()/n_known) if n_known else np.nan,
            'known_success_labels':int((mask & known & success).sum()),
            'known_success_reviewed':int((mask & known & success & reviewed).sum()),
            'unlabelled_events':int((mask & ~known).sum()),
            'unlabelled_alert_events':int((mask & ~known & alerted).sum()),
            'unlabelled_success_events':int(unknown_success.sum()),
            'unlabelled_success_alert_events':int((unknown_success & alerted).sum())}


def _case_counts(cases):
    reviewed = int(cases.admitted.sum())
    return {'cases':len(cases),'reviewed_cases':reviewed,'overflow_cases':len(cases)-reviewed}


def evaluate_alerts(truth, alerts, method, budgets=BUDGETS):

    if not truth.event_id.is_unique or not alerts.event_id.isin(truth.event_id).all():
        raise ValueError('Alert IDs must align with unique target records')
    prior = truth.set_index('event_id').has_prior_success
    alerted = truth.event_id.isin(alerts.event_id).to_numpy()
    rows,groups,daily,details,samples = [],[],[],[],[]
    for budget in budgets:
        label = 'unlimited' if budget is None else str(budget)
        cases,members = group_alerts(alerts,daily_budget=budget)
        cases['opening_has_prior_success'] = cases.first_event_id.map(prior).astype(bool)
        admitted = cases.loc[cases.admitted,'case_id']
        reviewed_ids = members.loc[members.case_id.isin(admitted),'event_id']
        reviewed = truth.event_id.isin(reviewed_ids).to_numpy()
        common = {'model':method,'daily_budget':label}
        counts = _record_counts(truth,alerted,reviewed,np.ones(len(truth),dtype=bool))
        rows.append({**common,**counts,**_case_counts(cases),
                     'repeated_alerts':len(members)-len(cases),
                     'compression_fraction':1-len(cases)/len(members) if len(members) else np.nan})
        for group,seen in [('no_prior_success',False),('has_prior_success',True)]:
            mask = truth.has_prior_success.eq(seen).to_numpy()
            group_cases = cases[cases.opening_has_prior_success.eq(seen)]
            groups.append({**common,'group':group,**_record_counts(truth,alerted,reviewed,mask),
                           **_case_counts(group_cases)})
        daily_cases = cases.assign(day=pd.to_datetime(cases.opened_at).dt.floor('D'))
        by_day = daily_cases.groupby('day').agg(cases=('case_id','size'),reviewed_cases=('admitted','sum'),
                                               member_alerts_by_open_day=('alert_count','sum')).reset_index()
        by_day['overflow_cases'] = by_day.cases-by_day.reviewed_cases
        by_day['model'],by_day['daily_budget'] = method,label
        daily.append(by_day)
        known_details = truth.loc[truth.known,['event_id','success','has_prior_success']].copy()
        known_details['alerted'] = alerted[truth.known.to_numpy()]
        known_details['reviewed'] = reviewed[truth.known.to_numpy()]
        known_details['case_id'] = known_details.event_id.map(members.set_index('event_id').case_id)
        known_details['model'],known_details['daily_budget'] = method,label
        details.append(known_details)
        sample = cases.head(10)[['case_id','opened_at','last_alert_at','first_event_id','alert_count',
                                'admitted','opening_has_prior_success']].copy()
        sample['model'],sample['daily_budget'] = method,label
        samples.append(sample)
    return (pd.DataFrame(rows),pd.DataFrame(groups),pd.concat(daily,ignore_index=True),
            pd.concat(details,ignore_index=True),pd.concat(samples,ignore_index=True))


class AlertCollector:
    def __init__(self, lock, start_row=8_000_001):
        self.lock,self.start_row = lock,start_row
        self.truth_parts = []
        self.alert_parts = {name:[] for name in METHODS}
        self.first_timestamp = None

    def observe(self, events, part):
        target = part['source_row'] >= self.start_row
        if not target.any():
            return
        if self.first_timestamp is None:
            self.first_timestamp = str(pd.Timestamp(part['time'][target][0]))
        self.truth_parts.append(pd.DataFrame({'event_id':part['source_row'][target],
                                              'known':part['known'][target], 'success':part['success'][target],
                                              'has_prior_success':part['prior'][target]>0}))
        scores = part | {'global_matched':part['global_surprisal'],
                         'personal_matched':part[self.lock['selected']]}
        for method in METHODS:
            flag = target & predictions(scores,method,self.lock['thresholds'][method])
            frame = events.loc[flag,['source_row','timestamp','user_id','ip_address']].copy()
            frame = frame.rename(columns={'source_row':'event_id'})
            frame['rule_names'],frame['model_alert'] = '',True
            frame['score'] = (part['forest'] if method.startswith('forest_') else scores[method])[flag]
            self.alert_parts[method].append(frame[ALERT_COLUMNS])

    def finish(self):
        truth = pd.concat(self.truth_parts,ignore_index=True)
        alerts = {name:pd.concat(parts,ignore_index=True) for name,parts in self.alert_parts.items()}
        return truth,alerts


def run(root=Path('.'), archive_path=Path('data/public/rba/rba-dataset.zip'),
        selection_path=None, reference_path=None):
    root = Path(root)
    source = resolve_run(root/'reports/rba_budget')
    selection_path = Path(selection_path) if selection_path is not None else source/'selection.json'
    reference_path = Path(reference_path) if reference_path is not None else source/'holdout_metrics.csv'
    with experiment_run(root/'reports/rba_triage','retrospective_triage') as output:
        selection_bytes = selection_path.read_bytes()
        lock = json.loads(selection_bytes)
        sha256,md5 = archive_hashes(archive_path)
        if md5!=PUBLISHER_MD5 or sha256!=lock['archive_sha256']:
            raise ValueError('Archive does not match the frozen public study')
        protocol = Path(__file__).parents[1]/'docs/EXPERIMENTS.md'
        config = {'study_type':'retrospective','budgets':list(BUDGETS),'methods':list(METHODS),
                  'window_seconds':600,'first_target_row':8_000_001,'last_target_row':12_000_000,
                  'selection_sha256':hashlib.sha256(selection_bytes).hexdigest(),
                  'reference_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                  'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
                  'archive_sha256':sha256,'archive_md5':md5,
                  'code_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ['rba_triage.py','rba_context.py','triage.py']}}
        (output/'selection.json').write_bytes(selection_bytes)
        (output/'protocol.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        collector = AlertCollector(lock)
        _,info = replay(archive_path,12_000_000,1_200_000,2_000_000,4_000_000,100_000,
                        extra_boundaries=(8_000_000,),observer=collector.observe,retain_scores=False)
        truth,alerts = collector.finish()
        results = [evaluate_alerts(truth,alerts[name],name) for name in METHODS]
        names = ['summary.csv','history_groups.csv','daily.csv','known_label_review.csv','case_sample.csv']
        tables = []
        for i,name in enumerate(names):
            table = pd.concat([result[i] for result in results],ignore_index=True)
            table.to_csv(output/name,index=False)
            tables.append(table)
        reference = pd.read_csv(reference_path).set_index('model')
        event_checks = {}
        for method in METHODS:
            row = tables[0][(tables[0].model==method)&(tables[0].daily_budget=='unlimited')].iloc[0]
            old = reference.loc[method]
            event_checks[method] = bool(row.events==old.events and row.alert_events==old.alerts and
                                         row.known_labels==old.known_positives and row.known_alerted==old.known_positives_alerted)
        if not all(event_checks.values()):
            raise ValueError('Event counts differ from the frozen reference; do not publish cases')
        info.update(study_type='retrospective',target_rows=len(truth),
                    first_target_timestamp=collector.first_timestamp,event_reference_checks=event_checks,
                    known_labels=int(truth.known.sum()),
                    known_failed_labels=int((truth.known & ~truth.success).sum()))
        (output/'audit.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
        print(tables[0].to_string(index=False),flush=True)
        return tables[0],info


if __name__ == '__main__':
    run()
