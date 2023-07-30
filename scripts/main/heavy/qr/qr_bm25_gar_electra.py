import pyterrier as pt
if not pt.started():
    pt.init()

import trec23
from trec23 import CONFIG, evaluate, copy_path
import os

from fire import Fire

import torch
import logging

def main(out_dir : str, irds : str = None, path : str = None, name : str = None, batch_size : int = 16, budget : int = 5000):
    assert irds is not None or path is not None, 'Either irds or path must be specified'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']

    ### LOAD MODEL ###

    logging.info('Loading model...')
    logging.info('Copying Corpus Graph...')
    corpus_graph_dir = copy_path(CONFIG['CORPUS_GRAPH_PATH'])
    logging.info('Corpus Graph Copied.')

    text_ref = pt.get_dataset('irds:msmarco-passage-v2')

    qr = trec23.load_qr(CONFIG['FLANT5_XXL_PATH'], llm_kwargs={'device_map' : 'sequential', 'load_in_8bit' : True, 'device' : device})
    bm25 = trec23.load_pisa(path='/tmp/msmarco-passage-v2-dedup.pisa', threads=4).bm25()
    electra = pt.text.get_text(text_ref, 'text') >> trec23.load_electra(CONFIG['ELECTRA_MARCO_PATH'], device=device, batch_size=batch_size, verbose=False)
    gar = trec23.load_gar(electra, corpus_graph_dir, num_results=budget, batch_size=batch_size, verbose=True)
    model = qr >> bm25 % budget >> gar

    logging.info('Done.')

    ### EVALUATE ###

    logging.info('Evaluating model...')
    evaluate(model, out_dir, irds, path, name)
    logging.info('Done.')
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)
        