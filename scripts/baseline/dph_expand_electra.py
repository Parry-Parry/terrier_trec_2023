import pyterrier as pt
if not pt.started():
    pt.init()

import trec23
from trec23 import CONFIG, evaluate
import os

from fire import Fire

import torch
import logging

def main(out_dir : str, irds : str = None, path : str = None, name : str = None, budget : int = 5000, qrels : str = None):
    assert irds is not None or path is not None, 'Either irds or path must be specified'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### LOAD MODEL ###

    logging.info('Loading model...')

    text_ref = pt.get_dataset('irds:msmarco-passage-v2')

    indexref = pt.IndexRef.of(CONFIG["TERRIER_MARCOv2_PATH"])
    dph = pt.BatchRetrieve(trec23.CONFIG["TERRIER_MARCOv2_PATH"], wmodel="DPH")
    dph_expand = dph >> pt.rewrite.Bo1QueryExpansion(indexref) >> dph % budget

    electra = trec23.load_electra(CONFIG['ELECTRA_MARCO_PATH'], device=device)
    model = dph_expand >> pt.apply.generic(lambda x : pt.model.pop_queries(x)) >>pt.text.get_text(text_ref, "text") >> electra

    logging.info('Done.')

    ### EVALUATE ###

    logging.info('Evaluating model...')
    evaluate(model, out_dir, irds, path, name, qrels)
    logging.info('Done.')
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)
        