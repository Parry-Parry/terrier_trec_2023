from pyterrier_adaptive import GAR, CorpusGraph
from trec23.pipelines.baselines import load_monot5
from typing import Optional

def load_gar(t5_model_name_or_path : str, 
             dataset : str, 
             graph_variant : str,
             k : int = 8,
             t5_kwargs : Optional[dict] = {}):
    
    monot5 = load_monot5(t5_model_name_or_path, **t5_kwargs)
    graph =  CorpusGraph.from_dataset(dataset, graph_variant).to_limit_k(k)

    return GAR(monot5, graph)