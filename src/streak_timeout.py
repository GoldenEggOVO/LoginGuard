from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import calculate_metrics
from .features import FEATURE_COLUMNS, build_features
from .generate_data import generate
from .mimicry import episode_stats
from .model import chronological_split, fit_with_threshold
from .rules import detect_rules

SEEDS = (149, 151, 157, 163, 167, 173, 179, 181)
TIMEOUTS = (('none', None), ('60s', 60), ('300s', 300))
GAPS = (20, 60, 150)


def validation_fpr_cutoff(scores, labels, budget=.02):
    scores, labels = np.asarray(scores), np.asarray(labels)
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels) or not 0 <= budget < 1:
        raise ValueError('Scores, labels, or FPR budget are invalid')
    normal = scores[labels == 0]
    if not len(normal) or not np.isfinite(scores).all():
        raise ValueError('Finite scores and normal validation events are required')
    allowed = int(np.floor(budget * len(normal)))
    boundary = np.sort(normal)[::-1][allowed]
    return float(np.nextafter(boundary, np.inf))


def run_study(seeds=SEEDS, root=Path('.')):
    output = Path(root) / 'reports' / 'streak_timeout'
    output.mkdir(parents=True, exist_ok=True)
    rows, episode_rows, attack_rows = [], [], []
    for seed in seeds:
        reference, truth, _ = generate(seed, test_profile=(20, True, False), include_episodes=True)
        train, validation, _ = chronological_split(reference.timestamp)
        y_validation = truth.loc[validation, 'is_anomaly'].to_numpy()
        normal_validation = y_validation == 0
        for timeout, seconds in TIMEOUTS:
            features = build_features(reference, failure_gap_seconds=seconds)
            model, cutoff_f1, _, _ = fit_with_threshold(features, train, validation, y_validation)
            val_scores = -model.score_samples(features.loc[validation, FEATURE_COLUMNS])
            cutoff_budget = validation_fpr_cutoff(val_scores, y_validation)
            validation_rule = (detect_rules(features.loc[validation]) != '').to_numpy()
            calibrations = [('fixed_rules', None, float(validation_rule[normal_validation].mean())),
                            ('validation_f1', cutoff_f1, float((val_scores[normal_validation] >= cutoff_f1).mean())),
                            ('validation_fpr_2pct', cutoff_budget, float((val_scores[normal_validation] >= cutoff_budget).mean()))]
            for gap in GAPS:
                events, labels, episodes = generate(seed, test_profile=(gap, True, False), include_episodes=True)
                test_features = build_features(events, failure_gap_seconds=seconds)
                _, _, test = chronological_split(events.timestamp)
                rules = (detect_rules(test_features) != '').to_numpy()
                scores = -model.score_samples(test_features.loc[:, FEATURE_COLUMNS])
                for calibration, cutoff, val_fpr in calibrations:
                    predicted = rules if cutoff is None else scores >= cutoff
                    test_truth = labels.loc[test, 'is_anomaly']
                    metrics = calculate_metrics(test_truth, predicted[np.asarray(test)])
                    ep_summary, ep_details = episode_stats(events, episodes, predicted.astype(int), test)
                    factors = {'seed': seed, 'timeout': timeout, 'gap_seconds': gap, 'calibration': calibration}
                    rows.append({**factors, 'threshold': cutoff, 'validation_fpr': val_fpr,
                                 'test_events': int(sum(test)), 'test_attacks': int(test_truth.sum()),
                                 **metrics, **ep_summary})
                    episode_rows.extend([{**factors, **row} for row in ep_details.to_dict('records')])
                    for attack_type in sorted(labels.loc[test & labels.is_anomaly.eq(1), 'attack_type'].unique()):
                        mask = np.asarray(test & labels.attack_type.eq(attack_type))
                        count, detected = int(mask.sum()), int(predicted[mask].sum())
                        attack_rows.append({**factors, 'attack_type': attack_type, 'attack_events': count,
                                            'detected': detected, 'recall': detected / count})
    results = pd.DataFrame(rows)
    results.to_csv(output / 'per_seed.csv', index=False)
    pd.DataFrame(episode_rows).to_csv(output / 'per_episode.csv', index=False)
    pd.DataFrame(attack_rows).to_csv(output / 'per_attack.csv', index=False)
    metrics = ['validation_fpr', 'precision', 'recall', 'f1', 'fpr', 'episode_recall', 'median_detected_seconds']
    summary = results.groupby(['timeout', 'gap_seconds', 'calibration'], sort=False)[metrics].agg(['mean', 'min', 'max'])
    summary.columns = ['_'.join(pair) for pair in summary.columns]
    summary.reset_index().to_csv(output / 'summary.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
    for col, calibration in enumerate(['validation_f1', 'validation_fpr_2pct']):
        for row, metric in enumerate(['recall', 'episode_recall']):
            ax = axes[row, col]
            for timeout, _ in TIMEOUTS:
                subset = results[(results.timeout == timeout) & (results.calibration == calibration)]
                values = subset.groupby('gap_seconds')[metric].mean()
                ax.plot(values.index, values.values, marker='o', label=f'Timeout: {timeout}')
            ax.set(title=f'{calibration}: {metric}', xticks=GAPS, ylim=(0, 1.05),
                   ylabel='Mean recall across seeds', xlabel='Attack-event interval (seconds)')
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)
    fig.suptitle('Synthetic detection versus streak inactivity timeout')
    fig.tight_layout(rect=(0, .07, 1, .96))
    fig.savefig(output / 'recall_by_timeout.png', dpi=160)
    plt.close(fig)
    return results


if __name__ == '__main__':
    frame = run_study()
    print(frame.groupby(['gap_seconds', 'timeout', 'calibration'])[['recall', 'episode_recall', 'fpr']].mean().round(4).to_string())
