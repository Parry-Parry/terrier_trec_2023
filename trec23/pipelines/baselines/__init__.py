import pyterrier as pt
if not pt.started():
    pt.init()
from typing import Any, Optional

def load_batchretrieve(index : Any, controls : Optional[dict] = None, properties : Optional[dict] = None, model : str = 'BM25'):
    return pt.BatchRetrieve(index, model=model, controls=controls, properties=properties)

def load_pisa(dataset : str = None, path : str = None, **kwargs):
    assert dataset is not None or path is not None, "Either dataset or path must be specified"
    from pyterrier_pisa import PisaIndex
    if path is not None:
        return PisaIndex(path, **kwargs)
    return PisaIndex.from_dataset(dataset, **kwargs)

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

def load_colbert(model_name_or_path : str, 
                 index_path : str, 
                 index_name : str, 
                 mode : str = 'e2e'):
    from pyterrier_colbert.ranking import ColBERTFactory
    pytcolbert = ColBERTFactory(model_name_or_path, index_path, index_name)

    return pytcolbert.text_scorer() if mode != 'e2e' else pytcolbert.end_to_end()

