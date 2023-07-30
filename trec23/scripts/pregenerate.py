from fire import Fire
import os
from trec23 import CONFIG
import ir_datasets as irds

def expansion(
        irds : str = None,
        path : str = None,
):
    assert irds is not None or path is not None, 'Either irds or path must be specified'
    if irds is not None:
        dataset = irds.load(irds)
        topics = dat
