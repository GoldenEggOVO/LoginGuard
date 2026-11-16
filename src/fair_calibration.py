"""Compare fixed and validation-calibrated rules on frozen new generator seeds."""

from itertools import product
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .evaluate import calculate_metrics
from .experiment_runs import experiment_run
from .features import FEATURE_COLUMNS, build_features
from .generate_data import generate
from .mimicry import episode_stats
from .model import chronological_split, fit_with_threshold
from .rules import detect_rules

SEEDS = (311, 313, 317, 331, 337, 347, 349, 353)
PROFILES = {'unfamiliar_fast': (20, False, False), 'primary_slow_shared': (150, True, True)}
RULE_GRID = tuple(product((2, 3, 4, 5, 6, 8, 10), (2, 3, 4, 5, 6, 8, 10), (1, 2, 3, 4, 5)))


def select_rule_thresholds(validation_features, validation_labels):
    """Search only the supplied validation slice; enumeration resolves F1 ties."""
    rows = []
    for failure, spray, success in RULE_GRID:
        limits = {'failure_limit': failure, 'spray_limit': spray, 'success_failure_limit': success}
        predicted = detect_rules(validation_features, **limits).ne('').to_numpy()
        rows.append({**limits, **calculate_metrics(validation_labels, predicted)})
    candidates = pd.DataFrame(rows)
    winner = candidates.iloc[int(np.argmax(candidates.f1.to_numpy()))]
    return {name: int(winner[name]) for name in ('failure_limit', 'spray_limit', 'success_failure_limit')}, candidates


def run(root=Path('.')):
    with experiment_run(Path(root) / 'reports/fair_calibration', 'synthetic') as output:
        reference, labels = generate(42)
        features = build_features(reference)
        train, validation, _ = chronological_split(reference.timestamp)
        selected, candidates = select_rule_thresholds(features.loc[validation], labels.loc[validation, 'is_anomaly'])
        model, cutoff, forest_candidates, forest_f1s = fit_with_threshold(
            features, train, validation, labels.loc[validation, 'is_anomaly'])
        joblib.dump(model, output / 'forest.joblib')
        validation_truth = labels.loc[validation, 'is_anomaly']
        validation_predictions = {'fixed_rules': detect_rules(features.loc[validation]).ne('').to_numpy(),
                                  'calibrated_rules': detect_rules(features.loc[validation], **selected).ne('').to_numpy(),
                                  'forest': -model.score_samples(features.loc[validation, FEATURE_COLUMNS]) >= cutoff}
        source = Path(__file__).resolve().parent
        protocol = source.parent / 'docs/fair_calibration_protocol.md'
        selection = {'reference_seed': 42, 'evaluation_seeds': list(SEEDS), 'profiles': PROFILES,
                     'rule_thresholds': selected, 'rule_candidates': len(candidates),
                     'forest_threshold': cutoff, 'forest_candidates': len(forest_candidates),
                     'forest_validation_f1': float(max(forest_f1s)),
                     'validation_metrics': {name: calculate_metrics(validation_truth, predicted)
                                            for name, predicted in validation_predictions.items()},
                     'reference_events_sha256': hashlib.sha256(reference.loc[train | validation].to_csv(index=False).encode()).hexdigest(),
                     'reference_labels_sha256': hashlib.sha256(labels.loc[train | validation].to_csv(index=False).encode()).hexdigest(),
                     'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
                     'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                       for name in ('fair_calibration.py', 'generate_data.py', 'features.py', 'model.py', 'rules.py')},
                     'model_sha256': hashlib.sha256((output / 'forest.joblib').read_bytes()).hexdigest()}
        candidates.to_csv(output / 'rule_validation_candidates.csv', index=False)
        selection_bytes = json.dumps(selection, indent=2).encode()
        (output / 'selection.json').write_bytes(selection_bytes)
        rows = []
        # New streams are generated only after the source model and both cutoffs are frozen.
        for seed in SEEDS:
            for profile, parameters in PROFILES.items():
                events, truth, episodes = generate(seed, test_profile=parameters, include_episodes=True)
                target = build_features(events)
                _, _, test = chronological_split(events.timestamp)
                predictions = {'fixed_rules': detect_rules(target).ne('').to_numpy(),
                               'calibrated_rules': detect_rules(target, **selected).ne('').to_numpy(),
                               'forest': -model.score_samples(target[FEATURE_COLUMNS]) >= cutoff}
                for method, predicted in predictions.items():
                    batch, _ = episode_stats(events, episodes, predicted, test)
                    rows.append({'seed': seed, 'profile': profile, 'method': method,
                                 'test_events': int(test.sum()), 'attack_events': int(truth.loc[test, 'is_anomaly'].sum()),
                                 **calculate_metrics(truth.loc[test, 'is_anomaly'], predicted[np.asarray(test)]), **batch})
        results = pd.DataFrame(rows)
        results.to_csv(output / 'per_seed.csv', index=False)
        summary = results.groupby(['profile', 'method'], sort=False)[['precision', 'recall', 'f1', 'fpr', 'episode_recall']].agg(['mean', 'min', 'max'])
        summary.columns = ['_'.join(pair) for pair in summary.columns]
        summary.reset_index().to_csv(output / 'summary.csv', index=False)
        assert (output / 'selection.json').read_bytes() == selection_bytes
        print(summary.to_string(), flush=True)
        return results


if __name__ == '__main__':
    run()
