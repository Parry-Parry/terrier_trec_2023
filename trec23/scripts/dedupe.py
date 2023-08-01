import pyterrier as pt
if not pt.started(): pt.init()
from fire import Fire
import os
from trec23 import CONFIG, MarcoDuplicator
import json 
import gzip 
from os.path import join

def dedupe_res(path : str, out : str, runname : str = None, cut : bool = False, budget : int = 2000):
    lookup = json.load(gzip.open(CONFIG['MARCOv2_DUPE_PATH'], 'rb'))
    res = pt.io.read_results(path)
    dupe = MarcoDuplicator(lookup)

    # group by qid, sort by rank and dedupe
    if cut: res = res.groupby('qid').apply(lambda x: x.sort_values('rank').head(100)).reset_index(drop=True)
    res = dupe.transform(res)

    if runname is not None:
        res['run_name'] = runname

    if cut: res = res.groupby('qid').apply(lambda x: x.sort_values('rank').head(100)).reset_index(drop=True)
    else: res = res.groupby('qid').apply(lambda x: x.sort_values('rank').head(budget)).reset_index(drop=True)
    res.sort_by(['qid', 'rank'], inplace=True)
    pt.io.write_results(res, join(out, f'deduped.{os.path.basename(path)}'), runname=runname)

if __name__ == '__main__':
    Fire(dedupe_res)