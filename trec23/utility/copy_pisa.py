from fire import Fire
from trec23 import CONFIG
import os 
import shutil

def copy_pisa(path : str = None) -> str:
    pisa_dir = CONFIG["PISA_MARCOv2_PATH"] if not path else path
    dir_name = os.path.basename(pisa_dir)
    new_dir = shutil.copytree(pisa_dir, os.path.join('tmp', dir_name))
    return new_dir

if __name__ == "__main__":
    Fire(copy_pisa)