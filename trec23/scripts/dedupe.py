import pyterrier as pt
pt.init()
from fire import Fire
import os
from trec23 import CONFIG, MarcoDuplicator
import json 
import gzip 

def dedupe_res(path : str, out : str):
    lookup = json.load(gzip.open(CONFIG['MARCOv2_DUPE_PATH'], 'rb'))
    res = pt.io.read_results(path)
    dupe = MarcoDuplicator(lookup)

    # group by qid, sort by rank and dedupe
    cut = res.groupby('qid').apply(lambda x: x.sort_values('rank').head(100)).reset_index(drop=True)
    new_res = dupe.transform(cut)

    pt.io.write_results(new_res, out)

if __name__ == '__main__':
    Fire(dedupe_res)