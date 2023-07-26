from fire import Fire
import os
from os.path import join
import subprocess as sp
import logging

def main(script_dir : str, out_dir : str, irds : str = None, path : str = None, budget : int = 5000):
    assert irds is not None or path is not None, 'Either irds or path must be specified'
    os.makedirs(out_dir, exist_ok=True)

    scripts = [f for f in os.listdir(script_dir)]

    for script in scripts:
        path = join(script_dir, script)
        name = script.strip('.py')
        args = f'python {path} --out_dir {join(out_dir, name)} --irds {irds} --path {path} --name {name} --budget {budget}'
        logging.info(f'Running {args}')
        sp.run(args, shell=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)