import pyterrier as pt

from fire import Fire
from trec23 import CONFIG, load_qr

def expansion(
        irds : str = None,
        path : str = None,
        out : str = None,
):
    assert irds is not None or path is not None, 'Either irds or path must be specified'
    if irds is not None:
        dataset = pt.get_dataset(irds)
        topics = dataset.get_topics()
    
    if path is not None:
        topics = pt.io.read_topics(path, format='singleline')

    qr = load_qr(CONFIG['FLANT5_XXL_PATH'], llm_kwargs={'device_map' : 'sequential', 'load_in_8bit' : True, 'device' : 'cuda'})

    expansions = qr.transform(topics)
    expansions.to_csv(out, sep='\t', header=False, index=False)

if __name__ == '__main__':
    Fire(expansion)
