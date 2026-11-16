import pandas as pd
from src.pipeline import run


def test_pipeline_writes_reproducible_outputs_without_label_features(tmp_path):
    run(tmp_path)
    data = pd.read_csv(tmp_path / 'data/login_events.csv')
    labels = pd.read_csv(tmp_path / 'data/ground_truth.csv')
    predictions = pd.read_csv(tmp_path / 'reports/predictions.csv')
    metrics = pd.read_csv(tmp_path / 'reports/metrics.csv')
    assert not {'is_anomaly', 'attack_type'} & set(data.columns)
    assert len(data) == len(labels) == len(predictions)
    assert set(predictions.split) == {'train', 'validation', 'test'}
    assert len(metrics) == 2
    assert all((tmp_path / 'reports/figures' / n).exists() for n in ['confusion_matrices.png', 'metrics_comparison.png', 'attack_recall.png'])
    original = predictions.copy()
    run(tmp_path)
    pd.testing.assert_frame_equal(original, pd.read_csv(tmp_path / 'reports/predictions.csv'))
