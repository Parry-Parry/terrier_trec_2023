
from fire import Fire
import os
from trec23 import CONFIG, MarcoDuplicator

def main(path : str, out : str):
    lookup = json.load(open(CONFIG['MARCOv2_DUPE_PATH'], 'r'))