import pyterrier as pt
if not pt.started():
    pt.init()

from trec23 import CONFIG, METRICS 
from trec23 import dual_experiment
import os
from os.path import join

from fire import Fire

def main(out_dir : str, irds : str = None, path : str = None, name : str = None):
    assert irds is not None or path is not None, 'Either irds or path must be specified'
    os.makedirs(out_dir, exist_ok=True)

    if irds is not None:
        ds = pt.get_dataset(irds)
        std, per_query = dual_experiment(model, names=[name], dataset=ds, metrics=METRICS)
        std.to_csv(join(out_dir, f'results.tsv'), sep='\t', index=False)
        per_query.to_csv(join(out_dir, f'perquery.tsv'), sep='\t', index=False)
    
    else:
        topics = pt.io.read_topics(path)
        results = model.transform(topics)
        pt.io.write_results(results, join(out_dir, f'{name}.trec'))
    
if __name__ == '__main__':
    Fire(main)
        