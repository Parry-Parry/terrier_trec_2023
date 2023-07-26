import pyterrier as pt
if not pt.started():
    pt.init()

import trec23
from trec23 import CONFIG, evaluate
import os

from fire import Fire

import torch
import logging

def main(out_dir : str, irds : str = None, path : str = None, name : str = None, budget : int = 5000):
    assert irds is not None or path is not None, 'Either irds or path must be specified'
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### LOAD MODEL ###

    logging.info('Loading model...')

    splade = trec23.load_splade(CONFIG['SPLADE_MARCOv2_PATH'], '/tmp/msmarco-passage-v2-dedup.splade.pisa', device=device)
    model = splade % budget

    logging.info('Done.')

    ### EVALUATE ###

    logging.info('Evaluating model...')
    evaluate(model, out_dir, irds, path, name)
    logging.info('Done.')
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)
        