from fire import Fire
import os
from os.path import join
import logging
import subprocess as sp

#irds = r'irds:msmarco-passage-v2/trec-dl-2022/judged'
#budget = '1000'
DIR = r'/resources/terrier_trec_2023/scripts/main/'

SUBDIRS = ['light', 'heavy/qr', 'heavy/prf']

def main(out_dir : str, path : str = None, irds : str = None, budget : int = 1000, use_cache=False, qrels : str = None, no_dl : bool = False):
    for d in SUBDIRS:
        script_dir = join(DIR, d)
        scripts = [f for f in os.listdir(script_dir) if not '__' in f]
        for script in scripts:
            spath = join(script_dir, script)
            name = script.strip('.py')
            if os.path.isdir(join(out_dir, name)):
                logging.info(f'Skipping {name} as it already exists in path {join(out_dir, name)}.')
                continue
            args = f'python {spath} --out_dir {join(out_dir, name)}  --name {name} --budget {budget}'
            args += f' --batch_size {64}'
            if irds: args += f' --irds {irds}'
            if path: args += f' --path {path}'
            if use_cache: args += ' --use_cache'
            if qrels: args += f' --qrels {qrels}'
            if no_dl: args += ' --no_dl'
            
            logging.info(f'Running {args}')
            sp.run(args, shell=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)