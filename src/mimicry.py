from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import calculate_metrics
from .features import FEATURE_COLUMNS, build_features
from .generate_data import generate
from .model import chronological_split, fit_with_threshold
from .rules import detect_rules

SEEDS = (103, 107, 109, 113, 127, 131, 137, 139)
PROFILES = tuple(product((20, 60, 150), (False, True), (False, True)))


def episode_stats(events, episodes, prediction, test):
    if not events.event_id.equals(episodes.event_id):
        raise ValueError('Episode metadata must align with event IDs')
    selected = pd.DataFrame({'episode_id': episodes.episode_id, 'timestamp': pd.to_datetime(events.timestamp),
                             'alert': np.asarray(prediction), 'test': np.asarray(test)})
    selected = selected[selected.test & selected.episode_id.notna()]
    rows = []
    for episode_id, group in selected.groupby('episode_id', sort=True):
        first_event = group.timestamp.min()
        alerted = group[group.alert == 1]
        detected = int(not alerted.empty)
        latency = (alerted.timestamp.min() - first_event).total_seconds() if detected else np.nan
        rows.append({'episode_id': episode_id, 'detected': detected, 'latency_seconds': latency})
    details = pd.DataFrame(rows, columns=['episode_id', 'detected', 'latency_seconds'])
    total, detected = len(details), int(details.detected.sum())
    summary = {'episodes': total, 'episodes_detected': detected, 'episodes_undetected': total - detected,
               'episode_recall': detected / total if total else np.nan,
               'median_detected_seconds': details.latency_seconds.median() if detected else np.nan}
    return summary, details


def run_study(seeds=SEEDS, root=Path('.')):
    output = Path(root) / 'reports' / 'mimicry'
    output.mkdir(parents=True, exist_ok=True)
    rows, episode_rows, attack_rows = [], [], []
    for seed in seeds:
        reference, reference_labels, _ = generate(seed, test_profile=PROFILES[0], include_episodes=True)
        reference_features = build_features(reference)
        train, validation, _ = chronological_split(reference.timestamp)
        model, cutoff, _, _ = fit_with_threshold(reference_features, train, validation,
                                                  reference_labels.loc[validation, 'is_anomaly'])
        for gap, familiar, shared in PROFILES:
            events, labels, episodes = generate(seed, test_profile=(gap, familiar, shared), include_episodes=True)
            features = build_features(events)
            _, _, test = chronological_split(events.timestamp)
            predictions = [('Rules', (detect_rules(features) != '').astype(int).to_numpy(), np.nan),
                           ('Isolation Forest', (-model.score_samples(features.loc[:, FEATURE_COLUMNS]) >= cutoff).astype(int), cutoff)]
            factors = {'seed': seed, 'gap_seconds': gap, 'familiar_device': familiar, 'shared_ip': shared}
            for method, predicted, threshold in predictions:
                test_truth = labels.loc[test, 'is_anomaly']
                metrics = calculate_metrics(test_truth, predicted[np.asarray(test)])
                ep_summary, ep_details = episode_stats(events, episodes, predicted, test)
                rows.append({**factors, 'method': method, 'threshold': threshold,
                             'test_events': int(sum(test)), 'test_attacks': int(test_truth.sum()),
                             **metrics, **ep_summary})
                episode_rows.extend([{**factors, 'method': method, **row} for row in ep_details.to_dict('records')])
                for attack_type in sorted(labels.loc[test & labels.is_anomaly.eq(1), 'attack_type'].unique()):
                    mask = np.asarray(test & labels.attack_type.eq(attack_type))
                    total, detected = int(mask.sum()), int(predicted[mask].sum())
                    attack_rows.append({**factors, 'method': method, 'attack_type': attack_type,
                                        'attack_events': total, 'detected': detected, 'recall': detected / total})
    results = pd.DataFrame(rows)
    results.to_csv(output / 'per_seed.csv', index=False)
    pd.DataFrame(episode_rows).to_csv(output / 'per_episode.csv', index=False)
    pd.DataFrame(attack_rows).to_csv(output / 'per_attack.csv', index=False)
    metrics = ['precision', 'recall', 'f1', 'fpr', 'episode_recall', 'median_detected_seconds']
    grouping = ['gap_seconds', 'familiar_device', 'shared_ip', 'method']
    summary = results.groupby(grouping, sort=False)[metrics].agg(['mean', 'min', 'max'])
    summary.columns = ['_'.join(pair) for pair in summary.columns]
    summary.reset_index().to_csv(output / 'summary.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, (familiar, shared) in zip(axes.flat, product((False, True), (False, True))):
        for method, marker in [('Rules', 'o'), ('Isolation Forest', 's')]:
            subset = results[(results.familiar_device == familiar) & (results.shared_ip == shared) & (results.method == method)]
            for metric, style in [('recall', '-'), ('episode_recall', '--')]:
                means = subset.groupby('gap_seconds')[metric].mean()
                ax.plot(means.index, means.values, marker=marker, linestyle=style, label=f'{method}: {metric}')
        ax.set(title=f'Familiar device: {familiar}; shared IP: {shared}', xlabel='Attack-event spacing (seconds)',
               ylabel='Mean recall across generator seeds', xticks=[20, 60, 150], ylim=(0, 1.05))
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9)
    fig.suptitle('Synthetic event and attack-batch detection')
    fig.tight_layout(rect=(0, .09, 1, .96))
    fig.savefig(output / 'recall_by_spacing.png', dpi=160)
    plt.close(fig)
    return results


if __name__ == '__main__':
    frame = run_study()
    print(frame.groupby(['gap_seconds', 'familiar_device', 'shared_ip', 'method'])[['recall', 'episode_recall', 'fpr']].mean().round(4).to_string())
