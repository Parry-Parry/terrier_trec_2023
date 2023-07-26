from pyterrier_adaptive import GAR, CorpusGraph
from trec23.pipelines.baselines import load_monot5
from typing import Optional, Any

def load_gar(model : Any, 
             dataset : str, 
             graph_variant : str,
             k : int = 8):
    
    graph =  CorpusGraph.from_dataset(dataset, graph_variant).to_limit_k(k)

    return GAR(model, graph)