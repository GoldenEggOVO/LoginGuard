from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import calculate_metrics
from .features import build_features
from .generate_data import generate
from .independent_data import generate_independent
from .triage import evaluate_cases, group_alerts
from .triage_study import METHODS, SEEDS, fit_reference, make_alerts


def run_study(seeds=SEEDS, budget=5, root=Path('.')):
    output = Path(root) / 'reports' / 'shift'
    output.mkdir(parents=True, exist_ok=True)
    model, cutoff = fit_reference()
    rows, diagnostics = [], []
    for seed in seeds:
        for name, generator in [('original', lambda s: generate(s, include_episodes=True)),
                                ('independent', generate_independent)]:
            events, labels, episodes = generator(seed)
            for method in METHODS:
                alerts, test = make_alerts(events, model, cutoff, method)
                test_events = events.loc[test].reset_index(drop=True)
                truth = labels.loc[test, 'is_anomaly'].to_numpy()
                predicted = test_events.event_id.isin(alerts.event_id).to_numpy()
                event_metrics = calculate_metrics(truth, predicted)
                if method == 'Isolation Forest':
                    test_features = build_features(events).loc[test].reset_index(drop=True)
                    normal = truth == 0
                    normal_alert = normal & predicted
                    failed = test_events.status.eq('failure').to_numpy()
                    prior = test_features.prior_consecutive_failures.gt(0).to_numpy()
                    diagnostics.append({'seed': seed, 'dataset': name, 'normal_events': int(normal.sum()),
                                        'alerted_normal_events': int(normal_alert.sum()),
                                        'normal_failure_rate': float(failed[normal].mean()),
                                        'normal_prior_failure_rate': float(prior[normal].mean()),
                                        'alerted_normal_failure_share': float(failed[normal_alert].mean()) if normal_alert.any() else np.nan,
                                        'alerted_normal_prior_failure_share': float(prior[normal_alert].mean()) if normal_alert.any() else np.nan})
                cases, members = group_alerts(alerts, daily_budget=budget)
                case_metrics, _ = evaluate_cases(test_events, episodes.loc[test].reset_index(drop=True), cases, members)
                rows.append({'seed': seed, 'dataset': name, 'method': method, 'daily_budget': budget,
                             'threshold': cutoff, 'test_events': len(test_events), 'test_attacks': int(truth.sum()),
                             **event_metrics, **case_metrics})
    results = pd.DataFrame(rows)
    results.to_csv(output / 'per_seed.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(output / 'normal_diagnostics.csv', index=False)
    metrics = ['precision', 'recall', 'f1', 'fpr', 'cases', 'reviewed_cases', 'episode_recall', 'label_relative_case_precision']
    summary = results.groupby(['dataset', 'method'], sort=False)[metrics].agg(['mean', 'min', 'max'])
    summary.columns = ['_'.join(pair) for pair in summary.columns]
    summary.reset_index().to_csv(output / 'summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, metric, title in zip(axes, ['recall', 'episode_recall'], ['Alert-event recall', 'Batch coverage with budget']):
        means = results.groupby(['dataset', 'method'])[metric].mean().unstack('dataset').reindex(METHODS)
        means.plot.bar(ax=ax, rot=15)
        ax.set(title=title, ylabel='Mean across generator seeds', xlabel='', ylim=(0, 1.05))
    fig.suptitle(f'Synthetic generators; first-come budget {budget} cases/day')
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)
    return results


if __name__ == '__main__':
    frame = run_study()
    print(frame.groupby(['dataset', 'method'])[['recall', 'fpr', 'episode_recall', 'cases']].mean().round(3).to_string())
