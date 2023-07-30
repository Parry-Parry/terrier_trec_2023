from fire import Fire
import os
from os.path import join
import logging
import subprocess as sp

irds = r'irds:msmarco-passage-v2/trec-dl-2021/judged'
budget = '1000'
DIR = r'/resources/terrier_trec_2023/scripts/main/'
OUT_DIR = r'/resources/TREC23/eval/main/dl22'

SUBDIRS = ['light', 'heavy/qr', 'heavy/prf']

def main():
    for d in SUBDIRS:
        script_dir = join(DIR, d)
        scripts = [f for f in os.listdir(script_dir) if not '__' in f]
        for script in scripts:
            spath = join(script_dir, script)
            name = script.strip('.py')
            if os.path.isdir(join(OUT_DIR, name)):
                logging.info(f'Skipping {name} as it already exists.')
                continue
            args = f'python {spath} --out_dir {join(OUT_DIR, name)}  --name {name} --budget {budget}'
            args += f' --batch_size {64}'
            args += f' --irds {irds}'
            
            logging.info(f'Running {args}')
            sp.run(args, shell=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)