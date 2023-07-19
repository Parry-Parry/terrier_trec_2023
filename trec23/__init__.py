import os 
import json 

try:
    with open(os.getenv("TREC_CONFIG", "config.json"), 'r') as f:
        try:
            CONFIG = json.load(f)
        except json.decoder.JSONDecodeError:
            CONFIG = {}
except FileNotFoundError:
    CONFIG = {}