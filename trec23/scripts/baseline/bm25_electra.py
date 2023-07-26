import pyterrier as pt
if not pt.started():
    pt.init()

import trec23
from trec23 import CONFIG, evaluate
import os

from fire import Fire

import torch

def main(out_dir : str, irds : str = None, path : str = None, name : str = None, budget : int = 5000):
    assert irds is not None or path is not None, 'Either irds or path must be specified'
    os.makedirs(out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ### LOAD MODEL ###

    index = pt.IndexFactory.of()
    text_ref = pt.Batchretrieve(CONFIG['TERRIER_MARCOv2_PATH'], metadata=['docno', 'text'])

    bm25 = trec23.load_pisa(path='/tmp/index.pisa').bm25()
    electra = trec23.load_electra(CONFIG['ELECTRA_BASE_PATH'], device=device)
    model = bm25 % budget >> pt.get_text(text_ref, 'body') >> electra

    ### EVALUATE ###

    evaluate(model, out_dir, irds, path, name)
    
if __name__ == '__main__':
    Fire(main)
        