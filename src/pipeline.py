from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from .generate_data import generate
from .features import build_features
from .rules import detect_rules
from .model import chronological_split, fit_and_score
from .evaluate import calculate_metrics, make_plots


def run(root=Path('.')):
    root = Path(root)
    (root / 'data').mkdir(parents=True, exist_ok=True)
    (root / 'reports').mkdir(parents=True, exist_ok=True)
    events, labels = generate()
    events.to_csv(root / 'data/login_events.csv', index=False)
    labels.to_csv(root / 'data/ground_truth.csv', index=False)
    features = build_features(events)
    train, val, test = chronological_split(events.timestamp)

    scores, cutoff, candidates, f1s = fit_and_score(features, train, val, labels.loc[val, 'is_anomaly'])
    rule_names = detect_rules(features)
    predictions = events[['event_id', 'timestamp']].copy()
    predictions['split'] = np.select([train, val, test], ['train', 'validation', 'test'], default='unknown')
    predictions['rule_names'] = rule_names
    predictions['rule_prediction'] = (rule_names != '').astype(int)
    predictions['anomaly_score'] = scores
    predictions['model_prediction'] = (scores >= cutoff).astype(int)
    predictions.to_csv(root / 'reports/predictions.csv', index=False)
    results = predictions.merge(labels, on='event_id', validate='one_to_one')
    test_results = results.loc[test].copy()
    rows = [{'method': name, **calculate_metrics(test_results.is_anomaly, test_results[col])}
            for name, col in [('Rules', 'rule_prediction'), ('Isolation Forest', 'model_prediction')]]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(root / 'reports/metrics.csv', index=False)
    test_results.groupby('attack_type')[['rule_prediction', 'model_prediction']].mean().to_csv(root / 'reports/scenario_recall.csv')
    val_y = labels.loc[val, 'is_anomaly']
    val_scores = scores[np.asarray(val)]
    pd.DataFrame([{'threshold': c, 'precision': precision_score(val_y, val_scores >= c, zero_division=0),
                   'recall': recall_score(val_y, val_scores >= c, zero_division=0), 'f1': f}
                  for c, f in zip(candidates, f1s)]).to_csv(root / 'reports/validation_thresholds.csv', index=False)
    details = events.merge(features, left_index=True, right_index=True).merge(results.drop(columns='timestamp'), on='event_id', validate='one_to_one')
    examples = []
    for method in ['rule', 'model']:
        for kind, truth, pred in [('false_positive', 0, 1), ('false_negative', 1, 0)]:
            subset = details[(details.split == 'test') & (details.is_anomaly == truth) & (details[method + '_prediction'] == pred)].head(5).copy()
            subset.insert(0, 'case', method + '_' + kind)
            examples.append(subset)
    pd.concat(examples).to_csv(root / 'reports/error_examples.csv', index=False)
    make_plots(test_results, metrics, root / 'reports/figures')
    (root / 'reports/threshold.txt').write_text(f'{cutoff:.17g}\n', encoding='utf-8')
    print(f'Events: {len(events)}; train/validation/test: {sum(train)}/{sum(val)}/{sum(test)}')
    print(f'Validation-selected anomaly threshold: {cutoff:.6f}')
    print(metrics.to_string(index=False))
    return metrics


if __name__ == '__main__':
    run()
