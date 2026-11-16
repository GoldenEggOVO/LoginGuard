import os
import hashlib
import json
from pathlib import Path
import tempfile

import pandas as pd
from src.experiment_runs import experiment_run, resolve_run


def run():
    base = Path('reports/rba_context')
    source = resolve_run(base)
    with experiment_run(base/'plots','plot') as output:
        (output/'inputs.json').write_text(json.dumps({name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                for name in ['holdout_metrics.csv','holdout_groups.csv']},indent=2),encoding='utf-8')
        _plot(source,output)


def _plot(source,output):
    metrics = pd.read_csv(source/'holdout_metrics.csv').set_index('model')
    groups = pd.read_csv(source/'holdout_groups.csv')
    names = metrics.index.tolist()
    labels = ['All-event\nforest','Successful\nforest','Global\ncontext','Personal\ncontext']
    colors = ['#8a96a6','#6882a1','#d08b4f','#247c75']
    with tempfile.TemporaryDirectory(prefix='loginguard-plot-') as config:
        os.environ['MPLCONFIGDIR'] = config
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','axes.unicode_minus':False,'font.size':10})
        fig, axes = plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
        values = metrics.known_positives_alerted.to_numpy()
        axes[0].bar(labels,values,color=colors)
        for i,row in enumerate(metrics.itertuples()):
            axes[0].text(i,row.known_positives_alerted+.35,
                         f'{row.known_positives_alerted}/{row.known_positives}',ha='center')
        axes[0].set(title='Known takeover records detected',ylabel='Record count',ylim=(0,27))
        values = metrics.unlabelled_success_alert_rate.to_numpy()*100
        axes[1].bar(labels,values,color=colors)
        for i,value in enumerate(values):
            axes[1].text(i,value+.04,f'{value:.2f}%',ha='center')
        axes[1].set(title='Alerts on unlabelled successes',ylabel='Percent (not FPR)',ylim=(0,3.2))
        for name,color,label in zip(names,colors,labels):
            sub = groups[(groups.model==name)&groups.group.isin(['early','late'])]
            axes[2].plot([0,1],sub.known_positive_recall*100,'o-',label=label,color=color)
        axes[2].set(xticks=[0,1],xticklabels=['First 2M (10 known)','Next 2M (14 known)'],
                    title='Known-positive discovery by period',ylabel='Percent',ylim=(-5,105))
        axes[2].legend(fontsize=8,loc='center right')
        for ax in axes[:2]:
            ax.tick_params(axis='x',labelrotation=20)
        for ax in axes:
            ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y',alpha=.15)
            ax.set_axisbelow(True)
        fig.suptitle('LoginGuard: 4M later records from one synthesized source; partial labels, not production performance',fontsize=12)
        fig.savefig(output/'holdout_comparison.png',dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    run()
