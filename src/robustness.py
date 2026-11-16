from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import calculate_metrics
from .features import FEATURE_COLUMNS, build_features
from .generate_data import generate
from .model import chronological_split, fit_and_score
from .rules import detect_rules

SEEDS = (11, 23, 37, 42, 53, 67, 79, 83, 97, 101)
SCENARIOS = ('baseline', 'familiar_slow')
METHODS = ('Rules', 'Isolation Forest', 'Isolation Forest without new_device')


def run_experiments(seeds=SEEDS, root=Path('.')):
    output = Path(root) / 'reports' / 'robustness'
    output.mkdir(parents=True, exist_ok=True)
    rows, attack_rows = [], []
    for seed in seeds:
        for scenario in SCENARIOS:
            events, labels = generate(seed, scenario=scenario)
            features = build_features(events)
            train, validation, test = chronological_split(events.timestamp)
            rule_predictions = (detect_rules(features) != '').astype(int).to_numpy()
            predictions = [(METHODS[0], rule_predictions, np.nan)]
            for method, columns in [(METHODS[1], FEATURE_COLUMNS),
                                    (METHODS[2], [c for c in FEATURE_COLUMNS if c != 'new_device'])]:
                scores, threshold, _, _ = fit_and_score(features, train, validation,
                                                         labels.loc[validation, 'is_anomaly'],
                                                         feature_columns=columns)
                predictions.append((method, (scores >= threshold).astype(int), threshold))
            y_test = labels.loc[test, 'is_anomaly']
            for method, predicted, threshold in predictions:
                rows.append({'seed': seed, 'scenario': scenario, 'method': method,
                             'test_events': int(sum(test)), 'test_attacks': int(y_test.sum()),
                             'threshold': threshold, **calculate_metrics(y_test, predicted[np.asarray(test)])})
                for attack_type in sorted(labels.loc[test & (labels.is_anomaly == 1), 'attack_type'].unique()):
                    mask = np.asarray(test & (labels.attack_type == attack_type))
                    count = int(mask.sum())
                    detected = int(predicted[mask].sum())
                    attack_rows.append({'seed': seed, 'scenario': scenario, 'method': method,
                                        'attack_type': attack_type, 'attack_events': count,
                                        'detected': detected, 'recall': detected / count})
    results = pd.DataFrame(rows)
    results.to_csv(output / 'per_seed.csv', index=False)
    pd.DataFrame(attack_rows).to_csv(output / 'per_attack.csv', index=False)
    metrics = ['precision', 'recall', 'f1', 'fpr', 'fnr']
    summary = results.groupby(['scenario', 'method'], sort=False)[metrics].agg(['mean', 'std', 'min', 'max'])
    summary.columns = ['_'.join(pair) for pair in summary.columns]
    summary.reset_index().to_csv(output / 'summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        subset = results[results.scenario == scenario]
        for index, method in enumerate(METHODS):
            values = subset.loc[subset.method == method, 'f1'].to_numpy()
            ax.scatter(np.full(len(values), index), values, alpha=.65, s=20)
            ax.plot(index, values.mean(), marker='_', color='black', markersize=18, markeredgewidth=2)
        ax.set(xticks=range(3), xticklabels=['Rules', 'IF', 'IF without\nnew device'], title=scenario,
               ylabel='Test event F1', ylim=(0, 1.05))
    fig.suptitle('Each dot is one generator seed; black mark is the mean')
    fig.tight_layout()
    fig.savefig(output / 'f1_by_seed.png', dpi=160)
    plt.close(fig)
    return results


if __name__ == '__main__':
    frame = run_experiments()
    print(frame.groupby(['scenario', 'method'], sort=False)[['precision', 'recall', 'f1', 'fpr']].mean().round(4).to_string())
