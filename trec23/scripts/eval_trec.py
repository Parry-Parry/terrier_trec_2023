import pyterrier as pt
if not pt.started(): pt.init()
from fire import Fire 
from ir_measures import *
import ir_measures 
import os 
from os.path import join
import json 

def main(run_dir : str, out_dir : str, dataset : str = None, qrels_path : str = None):
    assert dataset is not None or qrels_path is not None, 'Either dataset or qrels_path must be specified'
    METRICS = [nDCG@10, nDCG@100, R(rel=2)@100, R(rel=2)@1000, RR(rel=2)@10, AP(rel=2)@100, AP(rel=2)@1000, Judged@10, Judged@100, P(rel=2)@10]

    if qrels_path: 
        qrels = ir_measures.read_trec_qrels(qrels_path)
    else:
        dataset = pt.load_dataset(dataset)
    qrels = dataset.get_qrels()

    for path in os.listdir(run_dir):
        run = ir_measures.read_trec_run(join(run_dir, path))
        name = path.strip('.res.gz')
        res = ir_measures.calc_aggregate(METRICS, run, qrels)
        
        with open(join(out_dir, name + '.json'), 'w') as f:
            json.dump(res, f)

if __name__ == '__main__':
    Fire(main)


