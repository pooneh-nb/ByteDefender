
from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
import json

def get_bytecode_batches(bytecode):
    # initializa an empty list to store the batches
    batches = []

    # initialize valiable to keep track of the current batch
    current_batch = []
    in_batch = False

    # open the file and read it line by line
    for line in bytecode:
        # Check if the line contains the start marker
        if line.startswith("############### START ####################"):
            in_batch = True
            current_batch = []
        elif line.startswith("############### END ####################"):
            in_batch = False
            batches.append(current_batch)  # Add the batch to the list
        # If we're inside a batch, add the line to the current batch
        elif in_batch:
            current_batch.append(line)
    return batches

def main():
    output = Path.home().joinpath(Path.home().cwd(), "Test_single_sites/DB.json")
    log = Path.home().joinpath(Path.home().cwd(), "Test_single_sites/log.txt")
    dic = {}
    output_dir = Path.joinpath(Path.home().cwd(), 'server/output')
    sites = utilities.get_directories_in_a_directory(output_dir)
    invalid_urls = ["extensions::SafeBuiltins",
                    "v8/LoadTimes",
                    "chrome",
                    "extensions",
                    "file"]
                    # "chrome://",
                    # "extensions::",
                    # "chrome-extension://",
                    # "chrome-untrusted"clear
    for site in sites:
        try:
            siteName = site.split('/')[-1]
            files = utilities.get_files_in_a_directory(site)
            if len(files) == 3:
                # read contect of API traces and bytecodes
                for file in files:
                    if file.split('.')[-1] == 'json':
                        apiTraces = utilities.read_file_splitlines(file)
                    if file.split('/')[-1] == 'corrected_browser_logs.bin':
                        bytecode = utilities.read_file_newline_stripped(file)
                dic[siteName] = {}
                # analyze apiTraces.json
                for r in apiTraces:
                    record = json.loads(r)
                    if record["top_level_url"] == siteName:
                        scriptId = str(record["scriptId"])
                        scriptURL = record["scriptURL"]
                        functionName = record["functionName"]
                        if not any(scriptURL.startswith(invalid) for invalid in invalid_urls):
                            if record["scriptId"] not in dic[siteName]:
                                dic[siteName][scriptId] = {"scriptURL": scriptURL, "functions": {}}
                            if functionName != "":
                                if functionName not in dic[siteName][scriptId]["functions"]:
                                    dic[siteName][scriptId]["functions"][functionName] = {"API_args": [], "Bytecode": ""}
                                API = record["API"]
                                Args = ""
                                if "Args" in record:
                                    Args = record["Args"]
                                dic[siteName][scriptId]["functions"][functionName]["API_args"].append((API, Args))

                # analyze browser_logs.bin
                bc_batches = get_bytecode_batches(bytecode)
                for batch in bc_batches:
                    if batch != []:
                        script_url = batch[0].split('Script URL: ')[-1]
                        if not any(script_url.startswith(invalid) for invalid in invalid_urls):
                            try:
                                scriptId = batch[1].split('Script ID: ')[-1]
                                functionName = batch[2].split('Function name: ')[-1]
                                if functionName != 'Function name:':
                                    bytes = batch[3:]
                                    if scriptId in dic[siteName]:
                                        if functionName in dic[siteName][scriptId]["functions"]:
                                            dic[siteName][scriptId]["functions"][functionName]["Bytecode"] = bytes
                            except Exception as e:
                                print(e)
                                with open(log, 'a') as log_file:
                                    log_file.write(f"Site: {siteName}\n")
                                    log_file.write(f"Exception occurred: {str(e)}\n")
                                    log_file.write(f"Batch: {batch}\n")
                                    log_file.write("\n")  # Add a separator between entries
                                pass
        except Exception as e:
            print(e)
            with open(log, 'a') as file:
                file.write(f"Site: {siteName}\n")
                file.write(f"Exception occurred: {str(e)}\n")
            pass
            

    with open(output, 'w') as outfile:
        json.dump(dic, outfile, indent=4)
    
if __name__ == "__main__":
    main()