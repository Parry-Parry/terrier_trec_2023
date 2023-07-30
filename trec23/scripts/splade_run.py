import pyterrier as pt
pt.init()
import logging
import torch
from fire import Fire
from trec23 import CONFIG, copy_path, load_splade

def splade_run(topics_path : str, out_path : str, cutoff : int = 1000):
    topics = pt.io.read_topics(topics_path, format='singleline')
    logging.info("Copying PISA index & SPLADE...")
    pisa_dir = copy_path(CONFIG['PISA_SPLADE_PATH'])
    splade_dir = copy_path(CONFIG['SPLADE_MARCOv2_PATH'])

    splade = load_splade(splade_dir, pisa_dir, device=torch.device('cuda'))
    model = splade % cutoff

    logging.info("Running SPLADE...")
    res = model(topics)
    logging.info("Done.")

    res.to_csv(out_path, index=False, sep='\t')