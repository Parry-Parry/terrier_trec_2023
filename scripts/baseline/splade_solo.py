import pyterrier as pt
if not pt.started():
    pt.init()

import trec23
from trec23 import CONFIG, evaluate, copy_path
import os

from fire import Fire

import torch
import logging

def main(out_dir : str, irds : str = None, path : str = None, name : str = None, budget : int = 5000, qrels : str = None):
    assert irds is not None or path is not None, 'Either irds or path must be specified'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### LOAD MODEL ###

    logging.info('Loading model...')

    logging.info('Copying SPLADE...')
    splade_dir = copy_path(CONFIG['SPLADE_MARCOv2_PATH'])
    logging.info('SPLADE Copied.')

    _splade = trec23.load_splade(splade_dir, '/tmp/msmarco-passage-v2-dedup.splade.pisa', device=device)
    model = _splade % budget

    logging.info('Done.')

    ### EVALUATE ###

    logging.info('Evaluating model...')
    evaluate(model, out_dir, irds, path, name, qrels)
    logging.info('Done.')
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)
        