from pyterrier_adaptive import GAR, CorpusGraph
from typing import Optional, Any
import pdb

def load_gar(model : Any, 
             path : str, 
             k : int = None,
             **kwargs):
    pdb.set_trace()
    graph =  CorpusGraph.load(path)
    if k is not None:
        graph = graph.to_limit(k)
    return GAR(model, graph, **kwargs)