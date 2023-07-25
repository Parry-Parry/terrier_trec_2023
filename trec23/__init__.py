import os 
import json 
import logging

from . import runs
from . import utility
from . import evaluation

import pyterrier as pt
if not pt.started():
    pt.init()
from typing import Any, Optional

def load_batchretrieve(index : Any, 
                       controls : Optional[dict] = None, 
                       properties : Optional[dict] = None, 
                       model : str = 'BM25'):
    return pt.BatchRetrieve(index, model=model, controls=controls, properties=properties)

def load_pisa(dataset : str = None, 
              path : str = None, 
              **kwargs):
    assert dataset is not None or path is not None, "Either dataset or path must be specified"
    from pyterrier_pisa import PisaIndex
    if path is not None:
        return PisaIndex(path, **kwargs)
    return PisaIndex.from_dataset(dataset, **kwargs)

def load_splade(model_name_or_path : str, 
                pisa_path : str, 
                pisa_kwargs : dict = {}, 
                **kwargs):
    import pyt_splade
    mult = kwargs.pop("mult", 100.)
    pisa = load_pisa(path=pisa_path, **pisa_kwargs)
    splade = pyt_splade.SpladeFactory(model_name_or_path, **kwargs)
    return splade.query_encoder(scale=float(mult)) >> pisa.quantized()

def load_electra(model_name_or_path : str, **kwargs):
    from pyterrier_dr import ElectraScorer
    return ElectraScorer(model_name_or_path, **kwargs)

def load_monot5(model_name_or_path : str, **kwargs):
    from pyterrier_t5 import MonoT5ReRanker
    return MonoT5ReRanker(model=model_name_or_path, **kwargs)

def load_colbert(model_name_or_path : str, 
                 index_path : str, 
                 index_name : str, 
                 mode : str = 'e2e'):
    from pyterrier_colbert.ranking import ColBERTFactory
    pytcolbert = ColBERTFactory(model_name_or_path, index_path, index_name)

    return pytcolbert.text_scorer() if mode != 'e2e' else pytcolbert.end_to_end()

try:
    os.chdir('/')
    with open(os.getenv("TREC_CONFIG", "CONFIG.json"), 'r') as f:
        try:
            CONFIG = json.load(f)
        except json.decoder.JSONDecodeError:
            logging.warning("CONFIG.json file found, but could not be parsed. Using default configuration.")
            CONFIG = {}
except FileNotFoundError:
    logging.warning("No CONFIG.json file found. Using default configuration.")
    CONFIG = {}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(CONFIG)