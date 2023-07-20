import os 
import json 

try:
    with open(os.getenv("TREC_CONFIG", "CONFIG.json"), 'r') as f:
        try:
            CONFIG = json.load(f)
        except json.decoder.JSONDecodeError:
            CONFIG = {}
except FileNotFoundError:
    CONFIG = {}

if __name__ == '__main__':
    print(CONFIG)