from fire import Fire
import os
from os.path import join
import subprocess as sp
import logging
import trec23

def main(script_dir : str, 
         out_dir : str, 
         irds : str = None, 
         path : str = None, 
         qrels=None, 
         budget : int = 5000, 
         script_name : str = None, 
         batch_size : int = None, 
         use_cache=False,
         no_dl : bool = False,):
    assert irds is not None or path is not None, 'Either irds or path must be specified'
    os.makedirs(out_dir, exist_ok=True)

    scripts = [f for f in os.listdir(script_dir) if not '__' in f]

    # check if director '/tmp/index.pisa' exists   
    if not no_dl:
        if not os.path.exists('/tmp/msmarco-passage-v2-dedup.pisa'):
            logging.info('Copying PISA index...')
            trec23.copy_index(path=trec23.CONFIG["PISA_PATH"])
            logging.info('Done.')

        if not os.path.exists('/tmp/msmarco-passage-v2-dedup.splade.pisa'):
            logging.info('Copying PISA SPLADE index...')
            trec23.copy_index(path=trec23.CONFIG["PISA_SPLADE_PATH"])
            logging.info('Done.')

    if script_name is not None:
        scripts = [script_name]

    for script in scripts:
        spath = join(script_dir, script)
        name = script.strip('.py')
        if os.path.isdir(join(out_dir, name)):
            logging.info(f'Skipping {name} as it already exists.')
            continue
        args = f'python {spath} --out_dir {join(out_dir, name)}  --name {name} --budget {budget}'
        if batch_size: args += f' --batch_size {batch_size}'
        if irds: args += f' --irds {irds}'
        if path: args += f' --path {path}'
        if use_cache: args += ' --use_cache'
        if qrels: args += f' --qrels {qrels}'
        logging.info(f'Running {args}')
        sp.run(args, shell=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)