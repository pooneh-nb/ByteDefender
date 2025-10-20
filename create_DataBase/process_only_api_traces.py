
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

def label_fingerpriting_functions(db: Dict[str, Any], siteName: str):
    """
    Label functions as fingerpriting ot non-fingerpriting.

    :param dic: Dictionary to be updated.
    :param site_name: Name of the current site.
    """
    global canvas, audio, font, webrtc  # Declare the use of global variables
    for scriptURL, site_info in db[siteName].items():
        for scriptId, script_info in site_info.items():
            for functionName, func_info in script_info.items():
                api_args = func_info['API_args']
                if canvas_FP.is_tracking(api_args):
                    # canvas += 1
                    # print(f"canvas_FP: {canvas}, font_FP: {font}, audio_fp: {audio}, webrtc_fp: {webrtc}")
                    db[siteName][scriptURL][scriptId][functionName]["FP"] = True
                    return
                if font_FP.is_tracking(api_args):
                    # font += 1
                    # print(f"canvas_FP: {canvas}, font_FP: {font}, audio_fp: {audio}, webrtc_fp: {webrtc}")
                    db[siteName][scriptURL][scriptId][functionName]["FP"] = True
                    return
                if audioContext_FP.is_tracking(api_args):
                    # audio += 1
                    # print(f"canvas_FP: {canvas}, font_FP: {font}, audio_fp: {audio}, webrtc_fp: {webrtc}")
                    db[siteName][scriptURL][scriptId][functionName]["FP"] = True
                    return
                if webRTC_FP.is_tracking(api_args):
                    webrtc += 1
                    print(f"canvas_FP: {canvas}, font_FP: {font}, audio_fp: {audio}, webrtc_fp: {webrtc}")
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
            if "top_level_url" in record:
                if record["top_level_url"] == siteName:
                    if "scriptId" in record:
                        scriptId = str(record["scriptId"])
                        scriptURL = ""
                        if "scriptURL" in record:
                            scriptURL = record["scriptURL"]
                        if "functionName" in record:
                            functionName = record["functionName"]
                        if functionName == 'Function name:' or functionName == '':
                                functionName = generate_unique_function_name()

                        if not any(scriptURL.startswith(invalid) for invalid in INVALID_URLS):
                            site_dict = db.setdefault(siteName, {})
                            script_dict = site_dict.setdefault(scriptURL, {})
                            script_id_dict = script_dict.setdefault(scriptId, {})
                            function_dict = script_id_dict.setdefault(functionName, {"API_args": [], "FP": False, "Bytecode": ""})

                            # Append API and Args to the function dictionary if they exist.
                            API = record.get("API", "")
                            Args = record.get("Args", "")
                            if API or Args:
                                function_dict["API_args"].append((API, Args))
                            
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

def convert_db_to_dataframe() -> pd.DataFrame:
    print("convert_db_to_dataframe")
    # Prepare a list to store flattened data
    records = []

    db_path = Path(Path.cwd(), 'create_DataBase/apis/apis.json')
    df_output = Path(Path.cwd(), 'create_DataBase/apis/apis.csv')
    with open(db_path, 'r') as json_data:
        db = json.load(json_data)
    
    # Traverse the nested dictionary
    for siteName, site_data in tqdm(db.items()):
        for scriptURL, scripts_data in site_data.items():
            for scriptId, functions_data in scripts_data.items():
                for functionName, func_info in functions_data.items():
                    # Extract API_args, FP, Bytecode
                    API_args = func_info.get("API_args", [])
                    FP = func_info.get("FP", False)
                    Bytecode = func_info.get("Bytecode", "")

                    # Convert API_args to a more convenient representation if needed
                    # For example, you might store it as a JSON string
                    api_args_str = json.dumps(API_args)

                    # Create a record for each function
                    records.append({
                        "siteName": siteName,
                        "scriptURL": scriptURL,
                        "scriptId": scriptId,
                        "functionName": functionName,
                        "FP": FP,
                        "Bytecode": Bytecode,
                        "API_args": api_args_str
                    })

    # Convert the list of records into a DataFrame
    df = pd.DataFrame(records)
    df.to_csv(df_output, index=False)

def main(chunk, db):
    """
    Main function to process sites, API traces
    """
    CHUNK_NAME = chunk.split('/')[-1]
    # 
    print(CHUNK_NAME)

    
    log_path = Path(Path.cwd(), f"create_DataBase/apis/logs.txt")
    crawled_sites = utilities.get_directories_in_a_directory(chunk)
    
    for site in tqdm(crawled_sites):
        try:
            siteName = site.split('/')[-1]
            files = utilities.get_files_in_a_directory(site)
            apiTraces, raw_bytecode = None, None
            
            # read contect of API traces and bytecodes
            for file in files:
                if file.split('/')[-1] == 'apiTraces.json':
                    apiTraces = utilities.read_file_splitlines(file)
                
            if apiTraces:
                db[siteName] = {}
                # Process API traces
                process_api_traces(apiTraces, db, siteName, log_path)

                # label fingerpriting functions
                label_fingerpriting_functions(db, siteName)

            if db[siteName] in ('', [], None):
                del db[siteName]    
                
        except Exception as e:
            log_error(e, siteName, log_path, "main")
            # print("main", e)
            pass
            
    
if __name__ == "__main__":
    chunks = utilities.get_directories_in_a_directory(Path(Path.cwd(), 'server/crawled'))
    
    database_path = Path(Path.cwd(), f"create_DataBase/apis/apis.json")
    db = {}

    for chunk in chunks:
        print(len(db.keys()))
        main(chunk, db)
    
    with open(database_path, 'w') as outfile:
        json.dump(db, outfile, indent=4)

    convert_db_to_dataframe()
