from fire import Fire
import os 
import shutil

def copy_index(path : str) -> str:
    base = os.path.basename(path)
    new_dir = shutil.copytree(path, os.path.join('tmp', base))
    return new_dir

if __name__ == "__main__":
    Fire(copy_index)