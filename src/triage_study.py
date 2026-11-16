from html import escape
from pathlib import Path
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, build_features
from .generate_data import generate
from .model import chronological_split, fit_with_threshold
from .rules import detect_rules
from .triage import group_alerts, evaluate_cases

SEEDS = (241, 251, 257, 263, 269, 271, 277, 281)
BUDGETS = (2, 5, 10, None)
METHODS = ('Rules', 'Isolation Forest', 'Union')


def fit_reference():
    events, truth = generate(42)
    features = build_features(events)
    train, validation, _ = chronological_split(events.timestamp)
    model, cutoff, _, _ = fit_with_threshold(features, train, validation, truth.loc[validation, 'is_anomaly'])
    return model, cutoff


def make_alerts(events, model, cutoff, method):
    if method not in METHODS:
        raise ValueError('Unknown detector')
    features = build_features(events)
    _, _, test = chronological_split(events.timestamp)
    rule_names = detect_rules(features)
    scores = -model.score_samples(features.loc[:, FEATURE_COLUMNS])
    rule_alert = rule_names.ne('').to_numpy()
    model_alert = scores >= cutoff
    selected = {'Rules': rule_alert, 'Isolation Forest': model_alert, 'Union': rule_alert | model_alert}[method]
    frame = events.loc[test, ['event_id', 'timestamp', 'user_id', 'ip_address']].copy()
    frame['rule_names'] = rule_names.loc[test].to_numpy() if method != 'Isolation Forest' else ''
    frame['model_alert'] = model_alert[np.asarray(test)] if method != 'Rules' else False
    frame['score'] = scores[np.asarray(test)] if method != 'Rules' else np.nan
    return frame.loc[selected[np.asarray(test)]].reset_index(drop=True), test


def write_example(cases, members, output, seed, budget):
    cases.to_csv(output / 'example_cases.csv', index=False)
    members.to_csv(output / 'example_members.csv', index=False)
    columns = ['case_id', 'opened_at', 'user_id', 'ip_address', 'alert_count', 'rules', 'model_alert_seen', 'max_score', 'admitted']
    display = cases[columns].copy()
    display['admitted'] = display.admitted.map({True: 'Admitted', False: 'Over budget'})
    display['max_score'] = display.max_score.round(4)
    table = display.to_html(index=False, escape=True, na_rep='-', classes='cases')
    body = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>LoginGuard synthetic triage</title><style>body{font-family:system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;'
            'color:#17223b}table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border-bottom:1px solid #ccd4e2;'
            'padding:.5rem;text-align:left}th{background:#edf2fa}caption,p{line-height:1.5}.scroll{overflow-x:auto}</style></head><body>'
            '<h1>Synthetic alert triage</h1><p>Offline example; generated data only. '
            f'Seed {escape(str(seed))}, union detector, first-come daily budget {escape(str(budget))}. '
            'Admitted cases remain monitored for later alerts in their fixed ten-minute window. '
            'Rows show the completed window; admission used only information available when the case opened.</p>'
            f'<p>{len(cases)} cases · {int(cases.admitted.sum())} admitted · {len(cases) - int(cases.admitted.sum())} over budget</p>'
            '<div class="scroll">' + table + '</div></body></html>')
    (output / 'example.html').write_text(body, encoding='utf-8')


def run_study(seeds=SEEDS, budgets=BUDGETS, root=Path('.')):
    output = Path(root) / 'reports' / 'triage'
    output.mkdir(parents=True, exist_ok=True)
    model, cutoff = fit_reference()
    rows = []
    example_budget = 5 if 5 in budgets else budgets[0]
    for seed in seeds:
        events, _, metadata = generate(seed, include_episodes=True)
        for method in METHODS:
            alerts, test = make_alerts(events, model, cutoff, method)
            test_events = events.loc[test].reset_index(drop=True)
            test_meta = metadata.loc[test].reset_index(drop=True)
            for budget in budgets:
                cases, members = group_alerts(alerts, daily_budget=budget)
                metrics, _ = evaluate_cases(test_events, test_meta, cases, members)
                rows.append({'seed': seed, 'method': method, 'daily_budget': 'unlimited' if budget is None else str(budget),
                             'threshold': cutoff, **metrics})
                if seed == seeds[0] and method == 'Union' and budget == example_budget:
                    write_example(cases, members, output, seed, budget)
    results = pd.DataFrame(rows)
    results.to_csv(output / 'per_seed.csv', index=False)
    metrics = ['alert_events', 'cases', 'reviewed_cases', 'overflow_cases', 'repeated_alerts', 'episode_recall',
               'label_relative_case_precision', 'median_covered_latency_seconds']
    summary = results.groupby(['method', 'daily_budget'], sort=False)[metrics].agg(['mean', 'min', 'max'])
    summary.columns = ['_'.join(pair) for pair in summary.columns]
    summary.reset_index().to_csv(output / 'summary.csv', index=False)
    return results


if __name__ == '__main__':
    frame = run_study()
    print(frame.groupby(['method', 'daily_budget'])[['cases', 'reviewed_cases', 'episode_recall', 'label_relative_case_precision']].mean().round(3).to_string())
