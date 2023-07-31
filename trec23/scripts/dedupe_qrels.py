import pyterrier as pt
pt.init()
from fire import Fire 
import json
from trec23 import CONFIG
import gzip
from os.path import join

def main(dataset : str, out_dir : str):
    lookup = json.load(gzip.open(CONFIG['MARCOv2_DUPE_PATH'], 'rb'))
    super_lookup = []

    for _, v in lookup.items():
        for d in v:
            super_lookup.append(d)
    
    super_lookup = set(super_lookup)

    ds = pt.get_dataset(dataset)   
    qrels = ds.get_qrels()

    qrels = qrels[~qrels['docno'].isin(super_lookup)]
    qrels.to_csv(out_dir, sep='\t', index=False)

if __name__ == '__main__':
    Fire(main)