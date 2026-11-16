import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from .features import FEATURE_COLUMNS


def chronological_split(timestamps):
    t = pd.to_datetime(timestamps)
    start = t.min().normalize()
    return (t < start + pd.Timedelta(days=7),
            (t >= start + pd.Timedelta(days=7)) & (t < start + pd.Timedelta(days=10)),
            t >= start + pd.Timedelta(days=10))


def fit_with_threshold(features, train, validation, validation_labels, seed=42, feature_columns=None):
    """Fit without labels, then use validation labels to choose the alert cutoff."""
    columns = FEATURE_COLUMNS if feature_columns is None else list(feature_columns)
    if not columns or not set(columns).issubset(FEATURE_COLUMNS):
        raise ValueError('Features must come from the numeric feature allowlist')
    model = IsolationForest(n_estimators=200, contamination='auto', random_state=seed, n_jobs=1)
    model.fit(features.loc[train, columns])

    val_scores = -model.score_samples(features.loc[validation, columns])
    candidates = np.unique(np.quantile(val_scores, np.linspace(.80, .995, 80)))
    f1s = [f1_score(validation_labels, val_scores >= cutoff, zero_division=0) for cutoff in candidates]
    cutoff = float(candidates[int(np.argmax(f1s))])
    return model, cutoff, candidates, f1s


def fit_and_score(features, train, validation, validation_labels, seed=42, feature_columns=None):
    columns = FEATURE_COLUMNS if feature_columns is None else list(feature_columns)
    model, cutoff, candidates, f1s = fit_with_threshold(features, train, validation, validation_labels,
                                                         seed=seed, feature_columns=columns)
    scores = -model.score_samples(features.loc[:, columns])
    return scores, cutoff, candidates, f1s
