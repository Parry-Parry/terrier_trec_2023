from pyterrier_adaptive import GAR, CorpusGraph
from typing import Optional, Any

def load_gar(model : Any, 
             path : str, 
             k : int = None,
             **kwargs):
    graph =  CorpusGraph.load(path)
    if k is not None:
        graph = graph.to_limit(k)
    return GAR(model, graph, **kwargs)