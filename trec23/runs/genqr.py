from typing import Optional

def load_qr(llm_model_name_or_path : str, 
             llm_kwargs : Optional[dict] = {},
             **kwargs):
    from pyterrier_generativeqr.transformer import GenerativeQR 
    from pyterrier_generativeqr.models import FLANT5
    model = FLANT5(llm_model_name_or_path, **llm_kwargs)
    QR = GenerativeQR(model, **kwargs)

    return QR

def load_prf(llm_model_name_or_path : str, 
             llm_kwargs : Optional[dict] = {},
             **kwargs):
    from pyterrier_generativeqr.transformer import GenerativePRF 
    from pyterrier_generativeqr.models import FLANT5
    model = FLANT5(llm_model_name_or_path, **llm_kwargs)
    PRF = GenerativePRF(model, **kwargs)

    return PRF