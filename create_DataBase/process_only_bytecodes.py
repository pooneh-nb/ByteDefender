
from pathlib import Path
import json
from tqdm import tqdm
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
print(Path.cwd())
import utilities
from typing import List, Dict, Any
from fingerpriting_detector import audioContext_FP, canvas_FP, font_FP, webRTC_FP
import os
import time
import string
import random
import pandas as pd
import ijson
import csv


# Constants
START_MARKER = "############### START ####################"
END_MARKER = "############### END ####################"
BYTECODE_FILE = 'corrected_browser_logs.bin'
JSON_EXTENSION = 'json'
INVALID_URLS = ["extensions::SafeBuiltins", "v8/LoadTimes", "chrome", "extensions", "file"]


def generate_unique_function_name() -> str:
    timestamp = int(time.time() * 1000)  # Current time in milliseconds
    random_string = ''.join(random.choices(string.ascii_lowercase, k=5))  # 5 random lowercase letters
    function_name = f"anonymous_func_{timestamp}_{random_string}"
    return function_name


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

def process_bytecode_batches(batches: List[List[str]], db: Dict[str, Any], siteName: str, log_path: Path):
    """
    Processes bytecode batches and updates the dictionary.

    :param bc_batches: List of bytecode batches.
    :param dic: Dictionary to be updated.
    :param site_name: Name of the current site.
    :param log_path: Path to the log file.
    """
    for batch in batches:
        if batch != []:
            scriptURL = batch[0].split('Script URL: ')[-1].strip()
            if not any(scriptURL.startswith(invalid) for invalid in INVALID_URLS):
                try:
                    scriptId = batch[1].split('Script ID: ')[-1].strip()
                    functionName = batch[2].split('Function name: ')[-1].strip()
                    if functionName == 'Function name:' or functionName == '':
                        functionName = generate_unique_function_name()
                    bytecode = batch[4:]
                    if len(bytecode) == 5:
                        bytecode = batch[5:]
                    last_item_tokens = bytecode[-1].split(",")
                    bytes = bytecode[:-1] + last_item_tokens[:-1]
                    
                    site = db.setdefault(siteName, {})
                    scriptURLDict = site.setdefault(scriptURL, {})
                    scriptIdDict = scriptURLDict.setdefault(scriptId, {})

                    if functionName not in scriptIdDict:
                        scriptIdDict[functionName] = {"Bytecode": len(bytes) - 3}

                except Exception as e:
                    print(siteName, e)
                    log_error(e, siteName, log_path, "process_bytecode_batches", batch)
                    pass

def log_error(error: Exception, site_name: str, log_path: Path, source: str, batch: Any = None):
    """
    Logs errors to a specified log file.

    :param error: The exception to log.
    :param site_name: Name of the site where the error occurred.
    :param log_path: Path to the log file.
    :param batch: Optional batch data related to the error.
    """
    with open(log_path, 'a') as log_file:
        log_file.write(f"Site: {site_name}\n")
        log_file.write(f"Exception occurred: {str(error)}\n")
        log_file.write(f"source: {source}\n")
        if batch:
            log_file.write(f"{batch}\n")
        log_file.write("\n")

def convert_db_to_dataframe():

    db_path = Path(Path.cwd(), 'create_DataBase/bytecodes/bytecodes.json')
    df_output = Path(Path.cwd(), 'create_DataBase/bytecodes/bytecodes.csv')
    
    with open(db_path, 'r', encoding='utf-8') as f:
        # kvitems at the top-level will give us siteName and its corresponding object incrementally
        sites = ijson.kvitems(f, '')

        with open(df_output, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["siteName", "scriptURL", "scriptId", "functionName", "Bytecode"])
            writer.writeheader()

            for siteName, site_data in tqdm(sites):
                # site_data is a dictionary: { scriptURL: { scriptId: { functionName: { "Bytecode": ... }}}}
                # Process each nested level
                for scriptURL, scripts_data in site_data.items():
                    for scriptId, functions_data in scripts_data.items():
                        for functionName, func_info in functions_data.items():
                            Bytecode = func_info.get("Bytecode", None)
                            writer.writerow({
                                "siteName": siteName,
                                "scriptURL": scriptURL,
                                "scriptId": scriptId,
                                "functionName": functionName,
                                "Bytecode": Bytecode
                            })
    # df = pd.DataFrame(records)
    # df.to_csv(df, index=False)

def main(chunk, db):
    """
    Main function to process sites, API traces, and bytecode.
    """
    CHUNK_NAME = chunk.split('/')[-1]
    # 
    print(CHUNK_NAME)

    
    log_path = Path(Path.cwd(), f"create_DataBase/bytecodes/logs.txt")
    crawled_sites = utilities.get_directories_in_a_directory(chunk)
    
    for site in tqdm(crawled_sites):
        try:
            siteName = site.split('/')[-1]
            files = utilities.get_files_in_a_directory(site)
            bytecodes_path = Path.home().joinpath(site, 'bytecodes')
            
            raw_bytecodes = utilities.get_files_in_a_directory(bytecodes_path)
                
            if raw_bytecodes:
                db[siteName] = {}

                # Process bytecode batches
                batches = get_bytecode_batches(raw_bytecodes)
                process_bytecode_batches(batches, db, siteName, log_path)

            if db[siteName] in ('', [], None):
                del db[siteName]    
                
        except Exception as e:
            log_error(e, siteName, log_path, "main")
            # print("main", e)
            pass
               
if __name__ == "__main__":
    # chunks = utilities.get_directories_in_a_directory(Path(Path.cwd(), 'server/crawled'))
    
    # database_path = Path(Path.cwd(), f"create_DataBase/bytecodes/bytecodes.json")
    # db = {}

    # for chunk in chunks:
    #     print(len(db.keys()))
    #     main(chunk, db)
    
    # with open(database_path, 'w') as outfile:
    #     json.dump(db, outfile, indent=4)
    convert_db_to_dataframe()
