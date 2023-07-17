from typing import Optional
from trec23.pipelines.baselines import load_electra, load_pisa, load_splade
from pyterrier_generativeqr.transformer import GenerativeQR, GenerativePRF 
from pyterrier_generativeqr.models import FLANt5

def genqr_reranker(llm_model_name_or_path : str, 
                   electra_model_name_or_path : str, 
                   electra_kwargs : Optional[dict] = {}, 
                   llm_kwargs : Optional[dict] = {},
                   **kwargs):
    
    model = FLANt5(llm_model_name_or_path, **llm_kwargs)
    zeroshotQR = GenerativeQR(model, **kwargs)

    electra = load_electra(electra_model_name_or_path, **electra_kwargs)

    return zeroshotQR >> electra 