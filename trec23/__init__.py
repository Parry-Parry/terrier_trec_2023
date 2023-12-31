import os 
from os.path import join
import json 
import logging
from typing import Any, Optional
import shutil

import pyterrier as pt
from pyterrier.measures import *

METRICS = [nDCG@10, nDCG@100, R(rel=2)@100, R(rel=2)@1000, RR(rel=2)@10, AP(rel=2)@100, AP(rel=2)@1000, Judged@10, Judged@100, P(rel=2)@10]

def load_batchretrieve(index : Any, 
                       controls : Optional[dict] = None, 
                       properties : Optional[dict] = None, 
                       model : str = 'BM25'):
    return pt.BatchRetrieve(index, wmodel=model, controls=controls, properties=properties)

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

def evaluate(model, out_dir : str, irds : str, path : str, name : str, qrels : str = None):
    if not pt.started():
        pt.init()
    if irds is not None:
        ds = pt.get_dataset(irds)
        std = generate_experiment(model, names=[name], dataset=ds, metrics=METRICS, qrels=qrels)
        os.makedirs(out_dir, exist_ok=True)
        std.to_csv(join(out_dir, f'results.tsv'), sep='\t', index=False)
        #per_query.to_csv(join(out_dir, f'perquery.tsv'), sep='\t', index=False)
    else:
        topics = pt.io.read_topics(path, format='singleline')
        results = model.transform(topics)
        os.makedirs(out_dir, exist_ok=True)
        pt.io.write_results(results, join(out_dir, f'{name}.res.gz'))

def copy_path(path : str):
    base = os.path.basename(path)
    new_dir = os.path.join('/tmp', base)
    if not os.path.isdir(new_dir):
        new_dir = shutil.copytree(path, os.path.join('/tmp', base))
    return new_dir

def query_swap(df):
    df['query'] = df['query_0']
    df.drop(columns=['query_0'], inplace=True)
    return df

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

from .runs.duplicator import MarcoDuplicator
from .runs.gar import load_gar
from .runs.genqr import load_qr, load_prf
from .runs.h5cache import H5CacheScorer
from .evaluation import dual_experiment, generate_experiment
from .copy_pisa import copy_index

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(CONFIG)