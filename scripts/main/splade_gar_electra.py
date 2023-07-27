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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    
    ### LOAD MODEL ###

    logging.info('Loading model...')

    text_ref = pt.get_dataset('irds:msmarco-passage-v2')
    splade = trec23.load_splade(CONFIG['SPLADE_MARCOv2_PATH'], '/tmp/msmarco-passage-v2-dedup.splade.pisa', device=device)
    electra = pt.text.get_text(text_ref, 'text') >> trec23.load_electra(CONFIG['ELECTRA_MARCO_PATH'], device=device)
    gar = trec23.load_gar(electra, CONFIG['GAR_GRAPH_PATH'], num_results=budget, batch_size=16)
    model = splade % budget >> pt.text.get_text(text_ref, 'text') >> gar

    logging.info('Done.')

    ### EVALUATE ###

    logging.info('Evaluating model...')
    evaluate(model, out_dir, irds, path, name)
    logging.info('Done.')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)
        