
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

from multiprocessing import Pool

# Constants
START_MARKER = "############### START ####################"
END_MARKER = "############### END ####################"
BYTECODE_FILE = 'corrected_browser_logs.bin'
JSON_EXTENSION = 'json'
INVALID_URLS = ["extensions::SafeBuiltins", "v8/LoadTimes", "chrome", "extensions", "file", "No URL"]

canvas  = 0
audio = 0
font = 0
webrtc = 0

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
                    scriptId = batch[1].split('Script ID: ')[-1]
                    functionName = batch[2].split('Function name:')[-1].strip()
                    bytecode = batch[4:]
                    if len(bytecode) == 5:
                        bytecode = batch[5:]
                    last_item_tokens = bytecode[-1].split(",")
                    bytecodes = bytecode[:-1] + last_item_tokens[:-1]
                    site = db.setdefault(siteName, {})
                    script = site.setdefault(scriptURL, {})
                    functionDict = script.setdefault(scriptId, {})

                    if functionName:
                        current_entry = functionDict.get(functionName, {"API_args": [], "FP": False, "CallsAPI": False, "Bytecode": []})
                        current_entry["Bytecode"] = bytecodes
                        functionDict[functionName] = current_entry
                    else:
                        functionName = generate_unique_function_name()
                        fake_bytes = ['Parameter count 0', 'Register count 0', 'Frame size 0','anonymous_bytes']
                        current_entry = functionDict.get(functionName, {"API_args": [], "FP": False, "CallsAPI": False, "Bytecode": []})
                        current_entry["Bytecode"] = fake_bytes
                        functionDict[functionName] = current_entry

                except Exception as e:
                    print(siteName, e)
                    log_error(e, siteName, log_path, "process_bytecode_batches", batch)
                    pass

def label_fingerpriting_functions(db: Dict[str, Any], siteName: str):
    """
    Label functions as fingerpriting ot non-fingerpriting.

    :param dic: Dictionary to be updated.
    :param site_name: Name of the current site.
    """
    global canvas, audio, font, webrtc  # Declare the use of global variables
    for scriptURL, script_info in db[siteName].items():
        for scriptId, info in script_info.items():
            for functionName, func_info in info.items():
                api_args = func_info['API_args']
                if canvas_FP.is_tracking(api_args):
                    canvas += 1
                    # print(f"canvas_FP: {canvas}, font_FP: {font}, audio_fp: {audio}, webrtc_fp: {webrtc}")
                    db[siteName][scriptURL][scriptId][functionName]["FP"] = True
                    return
                if font_FP.is_tracking(api_args):
                    font += 1
                    # print(f"canvas_FP: {canvas}, font_FP: {font}, audio_fp: {audio}, webrtc_fp: {webrtc}")
                    db[siteName][scriptURL][scriptId][functionName]["FP"] = True
                    return
                if audioContext_FP.is_tracking(api_args):
                    audio += 1
                    # print(f"canvas_FP: {canvas}, font_FP: {font}, audio_fp: {audio}, webrtc_fp: {webrtc}")
                    db[siteName][scriptURL][scriptId][functionName]["FP"] = True
                    return
                if webRTC_FP.is_tracking(api_args):
                    webrtc += 1
                    # print(f"canvas_FP: {canvas}, font_FP: {font}, audio_fp: {audio}, webrtc_fp: {webrtc}")
                    db[siteName][scriptURL][scriptId][functionName]["FP"] = True
                    return


def process_api_traces(apiTraces: List[str], db: Dict[str, Any], siteName: str, log_path: Path):
    """
    Processes API traces and updates the dictionary.

    :param api_traces: List of API trace lines.
    :param dic: Dictionary to be updated.
    :param site_name: Name of the current site.
    :param log_path: Path to the log file.
    """
    for trace in apiTraces:
        record = json.loads(trace)   
        try:
            # Check if this record pertains to the current siteName and has necessary keys.
            if record.get("top_level_url") != siteName or "scriptId" not in record:
                continue

            # Retrieve necessary fields, with safe defaults.
            scriptId = str(record["scriptId"])
            scriptURL = record.get("scriptURL", "").strip()
            functionName = record.get("functionName", "")
            
            if not functionName:
                functionName = generate_unique_function_name()
            
            # Filter out invalid script URLs.
            if any(scriptURL.startswith(invalid) for invalid in INVALID_URLS):
                continue

            if scriptURL not in db[siteName]:
                db[siteName][scriptURL] = {}
            if scriptId not in db[siteName][scriptURL]:
                db[siteName][scriptURL][scriptId] = {}
            if functionName not in db[siteName][scriptURL][scriptId]:
                if functionName.startswith("anonymous_func_"):
                    # Update only the Bytecode to 'anonymous_bytes', keep existing API_args and FP
                    fake_bytes = ['Parameter count 0', 'Register count 0', 'Frame size 0','anonymous_bytes']
                    db[siteName][scriptURL][scriptId][functionName] = {"API_args": [], "FP": False, "CallsAPI": True, "Bytecode": fake_bytes}
                else:
                    # Regular update or insertion, set Bytecode as provided
                    db[siteName][scriptURL][scriptId][functionName] = {"API_args": [], "FP": False, "CallsAPI": True, "Bytecode": []}
            API = ""
            if "API" in record:
                API = record["API"]
            Args = ""
            if "Args" in record:
                Args = record["Args"]
            db[siteName][scriptURL][scriptId][functionName]["API_args"].append((API, Args))
                    
        except Exception as e:
            log_error(e, siteName, log_path, trace, "process_api_traces")
            print("process_api_traces", e)
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

def main(chunk):
    """
    Main function to process sites, API traces, and bytecode.
    """
    CHUNK_NAME = chunk.split('/')[-1]
    database_path = Path(Path.cwd(), f"create_DataBase/DB/script_level/DB_{CHUNK_NAME}.json")
    if not os.path.exists(database_path):
        print(CHUNK_NAME)
        log_path = Path(Path.cwd(), f"create_DataBase/DB/script_level/logs_DB/log_{CHUNK_NAME}.txt")
        crawled_sites = utilities.get_directories_in_a_directory(chunk)
        db = {}
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
                    db[siteName] = {}
                    # Process API traces
                    process_api_traces(apiTraces, db, siteName, log_path)

                    # label fingerpriting functions
                    label_fingerpriting_functions(db, siteName)

                    # Process bytecode batches
                    batches = get_bytecode_batches(raw_bytecodes)
                    process_bytecode_batches(batches, db, siteName, log_path)

                if db[siteName] in ('', [], None):
                    del db[siteName]    
                    
            except Exception as e:
                log_error(e, siteName, log_path, "main")
                # print("main", e)
                pass
                
        with open(database_path, 'w') as outfile:
            json.dump(db, outfile, indent=4)
        print(CHUNK_NAME, "is done!")
        db = {}
        crawled_sites = []
            
if __name__ == "__main__":
    chunks = utilities.get_directories_in_a_directory(Path(Path.cwd(), 'server/crawled'))
    
    # for chunk in chunks:
    #     main(chunk)

    # Create a pool of processes and map the function to the chunks
    with Pool(processes=3) as pool:
        pool.map(main, chunks)