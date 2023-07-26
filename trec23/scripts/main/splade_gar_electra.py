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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    
    ### LOAD MODEL ###

    text_ref = pt.Batchretrieve(CONFIG['TERRIER_MARCOv2_PATH'], metadata=['docno', 'text'])
    splade = trec23.load_splade(CONFIG['SPLADE_MARCOv2_PATH'], '/tmp/index.pisa')
    electra = trec23.load_electra(CONFIG['ELECTRA_MARCOv2_PATH'], device=device)
    model = splade % budget >> pt.get_text(text_ref, 'text') >> electra


    ### EVALUATE ###

    evaluate(model, out_dir, irds, path, name)
    
if __name__ == '__main__':
    Fire(main)
        