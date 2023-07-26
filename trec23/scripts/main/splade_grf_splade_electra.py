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
    devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']

    ### LOAD MODEL ###

    logging.info('Loading model...')

    text_ref = pt.Batchretrieve(CONFIG['TERRIER_MARCOv2_PATH'], metadata=['docno', 'text'])

    flan = trec23.load_flan(CONFIG['FLANT5_XXL_PATH'], device=devices[0], device_map='sequential', load_in_8bit=True)
    prf = trec23.load_prf(flan)
    splade = trec23.load_splade(CONFIG['SPLADE_MARCOv2_PATH'], '/tmp/index.pisa')
    electra = trec23.load_electra(CONFIG['ELECTRA_MARCOv2_PATH'], device=devices[1])
    splade_expand = splade % budget >> pt.get_text(text_ref, 'text') >> prf >> splade
    model = splade_expand >> pt.apply.generic(lambda x : pt.model.pop_queries(x))  >> electra

    logging.info('Done.')

    ### EVALUATE ###

    logging.info('Evaluating model...')
    evaluate(model, out_dir, irds, path, name)
    logging.info('Done.')
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)
        