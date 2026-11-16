from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from src.rba_validation import RAW_COLUMNS, FEATURES, adapt_rows, chronological_features


def run():
    targets = [(1,0,0,1,0),(1,0,0,1,1)]
    counts, state = Counter(), {}
    with ZipFile('data/public/rba/rba-dataset.zip') as archive, archive.open('rba-dataset.csv') as stream:
        for raw in pd.read_csv(stream,usecols=RAW_COLUMNS,dtype=str,keep_default_na=False,
                               chunksize=100_000,nrows=4_000_000):
            events,labels,_ = adapt_rows(raw.reset_index(drop=True))
            features = chronological_features(events,state)
            for target in targets:
                mask = (features.to_numpy() == target).all(axis=1)
                counts[(target,'events')] += int(mask.sum())
                counts[(target,'known_positive')] += int(labels.loc[mask,'label_state'].eq('known_positive').sum())
    output = Path('reports/rba_context/development_collisions.csv')
    rows = [{**dict(zip(FEATURES,target)), 'development_events':counts[(target,'events')],
             'known_positive_events':counts[(target,'known_positive')],
             'unlabelled_events':counts[(target,'events')]-counts[(target,'known_positive')]}
            for target in targets]
    result = pd.DataFrame(rows)
    result.to_csv(output,index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    run()
