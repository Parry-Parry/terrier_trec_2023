from fire import Fire
from trec23 import CONFIG
import os 
import shutil

def copy_index(path : str = None) -> str:
    pisa_dir = CONFIG["PISA_MARCOv2_PATH"] if not path else path
    new_dir = shutil.copytree(pisa_dir, os.path.join('tmp', 'index.pisa'))
    return new_dir

if __name__ == "__main__":
    Fire(copy_index)