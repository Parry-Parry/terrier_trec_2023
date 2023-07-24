import os 
import json 
import logging
try:
    os.chdir('/')
    with open(os.getenv("TREC_CONFIG", "CONFIG.json"), 'r') as f:
        try:
            CONFIG = json.load(f)
        except json.decoder.JSONDecodeError:
            logging.warning("CONFIG.json file found, but could not be parsed. Using default configuration.")
            CONFIG = {}
except FileNotFoundError:
    logging.warning("No CONFIG.json file found. Using default configuration.")
    CONFIG = {}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(CONFIG)