import os 
import json 

try:
    with open(os.getenv("TREC_CONFIG", "config.json"), 'r') as f:
        try:
            CONFIG = json.load(open(f, 'r'))
        except json.decoder.JSONDecodeError:
            CONFIG = {}
except FileNotFoundError:
    CONFIG = {}

if __name__ == '__main__':
    print(CONFIG)