from trec23.scripts.dedupe import dedupe_res
from fire import Fire 
import os

BASE = 'uofgtr_'

lookup = {
    'bm25_electra' : 'be',
    'bm25_gar_electra' : 'be_gb',
    'splade_gar_electra' : 'se_gb',
    'splade_electra' : 'se',
    'dph' : 'dph',
    'dph_bo1_dph' : 'dph_bo1',
    'dph_expand_electra' : 'dph_bo1_e',
    'splade_solo' : 's',
    'bm25_grf_electra' : 'b_grf_e',
    'bm25_grf_gar_electra' : 'b_grf_e_gb',
    'qr_bm25_electra' : 'qr_be',
    'qr_bm25_gar_electra' : 'qr_be_gb',
}

def main(parent_dir : str, out_dir, cut : bool = False, budget : int = 2000):
    # find all files in parent_dir and subdirs with extension .gz
    # for each file, run expansion on it
    # save to out_dir
    os.makedirs(out_dir, exist_ok=True)
    for root, _, files in os.walk(parent_dir):
        for file in files:
            if file.endswith('.gz'):
                name = os.path.basename(file).strip('.res.gz')
                try:
                    runname = BASE + lookup[name]
                except KeyError:
                    print(f'No lookup for {name}')
                    continue
                path = os.path.join(root, file)
                dedupe_res(path, out_dir, runname, cut=cut, budget=budget)

if __name__ == '__main__':
    Fire(main)