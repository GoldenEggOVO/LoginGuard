import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def calculate_metrics(truth, prediction):
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
            'precision': precision_score(truth, prediction, zero_division=0),
            'recall': recall_score(truth, prediction, zero_division=0),
            'f1': f1_score(truth, prediction, zero_division=0),
            'fpr': fp / (fp + tn) if fp + tn else 0., 'fnr': fn / (fn + tp) if fn + tp else 0.}


def make_plots(results, metrics, output):
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (name, col) in zip(axes, [('Rules', 'rule_prediction'), ('Isolation Forest', 'model_prediction')]):
        matrix = confusion_matrix(results.is_anomaly, results[col], labels=[0, 1])
        ax.imshow(matrix, cmap='Blues')
        for (i, j), v in np.ndenumerate(matrix):
            ax.text(j, i, str(v), ha='center', va='center', color='white' if v > matrix.max() / 2 else 'black')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'], xlabel='Predicted', ylabel='Actual', title=name)
    fig.tight_layout(); fig.savefig(output / 'confusion_matrices.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics.set_index('method')[['precision', 'recall', 'f1']].plot.bar(ax=ax, rot=0)
    ax.set_ylim(0, 1); ax.set_ylabel('Score'); fig.tight_layout(); fig.savefig(output / 'metrics_comparison.png', dpi=160); plt.close(fig)
    subset = results[results.is_anomaly == 1]
    by_attack = subset.groupby('attack_type')[['rule_prediction', 'model_prediction']].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    by_attack.plot.bar(ax=ax, rot=15); ax.set_ylabel('Event recall'); ax.set_ylim(0, 1)
    ax.legend(['Rules', 'Isolation Forest'], loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=2)
    fig.tight_layout(); fig.savefig(output / 'attack_recall.png', dpi=160); plt.close(fig)
