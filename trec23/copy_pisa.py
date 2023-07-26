from fire import Fire
from trec23 import CONFIG
import os 
import shutil

def copy_index(path : str = None, type="PISA_MARCOv2_PATH") -> str:
    pisa_dir = CONFIG[type] if not path else path
    base = os.path.basename(pisa_dir)
    new_dir = shutil.copytree(pisa_dir, os.path.join('tmp', base))
    return new_dir

if __name__ == "__main__":
    Fire(copy_index)