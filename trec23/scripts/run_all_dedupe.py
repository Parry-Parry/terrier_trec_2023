from .dedupe import dedupe_res
from fire import Fire 
import os

BASE = 'uofgtr_'

lookup = {

}

def main(parent_dir : str, out_dir, cut : bool = False):
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
                dedupe_res(path, out_dir, runname, cut=cut)

if __name__ == '__main__':
    Fire(main)