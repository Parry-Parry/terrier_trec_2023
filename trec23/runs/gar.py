from pyterrier_adaptive import GAR, CorpusGraph
from typing import Optional, Any

def load_gar(model : Any, 
             dataset : str, 
             graph_variant : str,
             k : int = 8):
    graph =  CorpusGraph.from_dataset(dataset, graph_variant).to_limit_k(k)

    return GAR(model, graph)