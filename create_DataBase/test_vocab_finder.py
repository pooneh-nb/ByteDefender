from pathlib import Path
import json
from tqdm import tqdm
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
print(Path.cwd())
import utilities
from typing import List, Dict, Any
import os


# Constants
START_MARKER = "############### START ####################"
END_MARKER = "############### END ####################"
BYTECODE_FILE = 'corrected_browser_logs.bin'
JSON_EXTENSION = 'json'
INVALID_URLS = ["extensions::SafeBuiltins", "v8/LoadTimes", "chrome", "extensions", "file", "No URL"]

JUMBLING_WORDS = ['############### START ####################', 'Script URL:', 'Script ID:', 
                  'Function name:', 'Bytecode:', '############### END ####################']


def get_bytecode_batches(bytecodes: List[str]) -> List[List[str]]:
    """
    Extracts bytecode batches from a list of lines.
    
    :param bytecode: List of lines from the bytecode file.
    :return: List of bytecode batches.
    """
    batches = []
    current_batch = []
    in_batch = False

    for bytecode_file in bytecodes:
        bytecode = utilities.read_file_newline_stripped(bytecode_file)
        for line in bytecode:
            if line.startswith(START_MARKER):
                in_batch = True
                current_batch = []
            elif line.startswith(END_MARKER):
                in_batch = False
                batches.append(current_batch)  
            elif in_batch:
                current_batch.append(line)
    return batches

def process_bytecode_batches(batches, siteName, vocab):
    """
    Processes bytecode batches and updates the dictionary.

    :param bc_batches: List of bytecode batches.
    :param dic: Dictionary to be updated.
    :param site_name: Name of the current site.
    :param log_path: Path to the log file.
    """
    new_vocab = set()
    vocabs  = list(vocab.keys())
    for batch in batches:
        if batch != []:
            scriptURL = batch[0].split('Script URL: ')[-1].strip()
            if not any(scriptURL.startswith(invalid) for invalid in INVALID_URLS):
                try:
                    scriptId = batch[1].split('Script ID: ')[-1]
                    functionName = batch[2].split('Function name:')[-1].strip()
                    bytecode = batch[4:]
                    if len(bytecode) == 5:
                        bytecode = batch[5:]
                    last_item_tokens = bytecode[-1].split(",")
                    bytecodes = bytecode[:-1] + last_item_tokens[:-1]

                    for bytes in bytecodes:
                        if bytes.strip() and not any(jumbling in bytes for jumbling in JUMBLING_WORDS):
                            if not (bytes.startswith("Parameter count") or bytes.startswith("Register count") or bytes.startswith("Frame size")):
                                if bytes not in vocabs:
                                    new_vocab.add(bytes)
                                    print(new_vocab)
                                    print("====================")
                    

                    
                except Exception as e:
                    print(siteName, e)
                    pass

def main(chunk):
    """
    Main function to process sites, API traces, and bytecode.
    """
    CHUNK_NAME = chunk.split('/')[-1]
    vocab = utilities.read_json(Path(Path.cwd(), 'create_DataBase/DB/script_level/dataset/vocab.json'))
    print(CHUNK_NAME)

    crawled_sites = utilities.get_directories_in_a_directory(chunk)

    for site in tqdm(crawled_sites):
        try:
            siteName = site.split('/')[-1]
            files = utilities.get_files_in_a_directory(site)
            bytecodes_path = Path.home().joinpath(site, 'bytecodes')
            apiTraces, raw_bytecodes = None, None
            
            # read contect of API traces and bytecodes
            for file in files:
                if file.split('/')[-1] == 'apiTraces.json':
                    apiTraces = utilities.read_file_splitlines(file)
            raw_bytecodes = utilities.get_files_in_a_directory(bytecodes_path)
                
            if apiTraces and raw_bytecodes:

                # Process bytecode batches
                batches = get_bytecode_batches(raw_bytecodes)
                process_bytecode_batches(batches, siteName, vocab)

                
        except Exception as e:
            print("main", e)
            pass

            
if __name__ == "__main__":
    chunks = utilities.get_directories_in_a_directory(Path(Path.cwd(), 'server/crawled'))
    
    for chunk in chunks:
        main(chunk)