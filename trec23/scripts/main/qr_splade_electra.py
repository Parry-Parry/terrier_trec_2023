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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']

    ### LOAD MODEL ###
    
    logging.info('Loading model...')

    text_ref = pt.Batchretrieve(CONFIG['TERRIER_MARCOv2_PATH'], metadata=['docno', 'text'])

    flan = trec23.load_flan(CONFIG['FLANT5_XXL_PATH'], device=devices[0], device_map='sequential', load_in_8bit=True)
    qr = trec23.load_qr(flan)

    splade = trec23.load_splade(CONFIG['SPLADE_MARCOv2_PATH'], '/tmp/msmarco-passage-v2-dedup.splade.pisa', device=devices[1])
    electra = trec23.load_electra(CONFIG['ELECTRA_MARCOv2_PATH'], device=devices[1])
    model = qr >> splade % budget >> pt.get_text(text_ref, 'text') >> electra

    logging.info('Done.')

    ### EVALUATE ###

    logging.info('Evaluating model...')
    evaluate(model, out_dir, irds, path, name)
    logging.info('Done.')
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)
        