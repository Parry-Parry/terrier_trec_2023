import pyterrier as pt
if not pt.started():
    pt.init()
from typing import Any

def load_batchretrieve(index : Any, controls=None, properties=None, model='BM25'):
    return pt.BatchRetrieve(index, model=model, controls=controls, properties=properties)

def load_pisa(dataset : str):
    from pyterrier_pisa import PisaIndex
    return PisaIndex.from_dataset(dataset)

def load_splade(model_name_or_path : str, **kwargs):
    import pyt_splade
    mult = kwargs.pop("mult", 100)
    splade = pyt_splade.SpladeFactory(model_name_or_path, **kwargs)
    return splade.query(mult=mult)

def load_electra(model_name_or_path, **kwargs):
    from pyterrier_dr import ElectraScorer
    return ElectraScorer(model_name_or_path, **kwargs)

def dph_bo1_dph(index):
    dph = load_batchretrieve(index, model="DPH")
    return dph >> pt.rewrite.Bo1QueryExpansion(index) >> dph

def bm25_electra(model_name_or_path, dataset : str = None, index : Any = None, bm25_kwargs : dict = {}, electra_kwargs : dict = {}):
    if dataset is not None:
        bm25 = load_pisa(dataset).bm25(**bm25_kwargs)
    else:
        bm25 = load_batchretrieve(index, model="BM25", **bm25_kwargs)

    return bm25 >> load_electra(model_name_or_path, **electra_kwargs)

def splade_electra(splade_model_name_or_path, electra_model_name_or_path, splade_kwargs={}, electra_kwargs={}):
    splade = load_splade(splade_model_name_or_path, **splade_kwargs)
    electra = load_electra(electra_model_name_or_path, **electra_kwargs)
    return splade >> electra

def dph_bo1_dph_electra(index, electra_model_name_or_path, electra_kwargs={}):
    return dph_bo1_dph(index) >> load_electra(electra_model_name_or_path, **electra_kwargs)


