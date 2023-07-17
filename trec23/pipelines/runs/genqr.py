from typing import Optional
from pyterrier_generativeqr.transformer import GenerativeQR, GenerativePRF 
from pyterrier_generativeqr.models import FLANt5

def load_qr(llm_model_name_or_path : str, 
             llm_kwargs : Optional[dict] = {},
             **kwargs):
    
    model = FLANt5(llm_model_name_or_path, **llm_kwargs)
    zeroshotQR = GenerativeQR(model, **kwargs)

    return zeroshotQR