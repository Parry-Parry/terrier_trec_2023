import pyterrier as pt
if not pt.started():
    pt.init()
from typing import Any, Optional

def load_batchretrieve(index : Any, controls : Optional[dict] = None, properties : Optional[dict] = None, model : str = 'BM25'):
    return pt.BatchRetrieve(index, model=model, controls=controls, properties=properties)

def load_pisa(dataset : str = None, path : str = None):
    assert dataset is not None or path is not None, "Either dataset or path must be specified"
    from pyterrier_pisa import PisaIndex
    if path is not None:
        return PisaIndex(path)
    return PisaIndex.from_dataset(dataset)

def load_splade(model_name_or_path : str, **kwargs):
    import pyt_splade
    mult = kwargs.pop("mult", 100)
    splade = pyt_splade.SpladeFactory(model_name_or_path, **kwargs)
    return splade.query(mult=mult)

def load_electra(model_name_or_path : str, **kwargs):
    from pyterrier_dr import ElectraScorer
    return ElectraScorer(model_name_or_path, **kwargs)

def load_monot5(model_name_or_path : str, **kwargs):
    from pyterrier_t5 import MonoT5ReRanker
    return MonoT5ReRanker(model=model_name_or_path, **kwargs)

def dph_bo1_dph(index):
    dph = load_batchretrieve(index, model = "DPH")
    return dph >> pt.rewrite.Bo1QueryExpansion(index) >> dph

def load_colbert(model_name_or_path : str, 
                 index_path : str, 
                 index_name : str, 
                 mode : str = 'e2e'):
    from pyterrier_colbert.ranking import ColBERTFactory
    pytcolbert = ColBERTFactory(model_name_or_path, index_path, index_name)

    return pytcolbert.text_scorer() if mode != 'e2e' else pytcolbert.end_to_end()

def bm25_electra(model_name_or_path : str, 
                 dataset : str = None, 
                 index : Any = None, 
                 cut : int = 100, 
                 bm25_kwargs : Optional[dict] = {}, 
                 electra_kwargs : Optional[dict] = {}):
    if dataset is not None:
        bm25 = load_pisa(dataset).bm25(**bm25_kwargs)
    else:
        bm25 = load_batchretrieve(index, model="BM25", **bm25_kwargs)

    return bm25 % cut >> pt.text.get_text(index, 'text') >> load_electra(model_name_or_path, **electra_kwargs)

def splade_electra(splade_model_name_or_path : str, 
                   electra_model_name_or_path : str, 
                   index, 
                   cut : int = 100, 
                   splade_kwargs : Optional[dict] = {}, 
                   electra_kwargs : Optional[dict] = {}):
    splade = load_splade(splade_model_name_or_path, **splade_kwargs)
    electra = load_electra(electra_model_name_or_path, **electra_kwargs)
    return splade % cut >> pt.text.get_text(index, 'text') >> electra

def dph_bo1_dph_electra(index : Any, electra_model_name_or_path : str, cut : int = 100, electra_kwargs : Optional[dict] = {}):
    return dph_bo1_dph(index) % cut >> pt.text.get_text(index, 'text') >> load_electra(electra_model_name_or_path, **electra_kwargs)