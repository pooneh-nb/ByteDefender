from ast import List
from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
import shutil
import json

INVALID_URLS = ["extensions::SafeBuiltins", "v8/LoadTimes", "chrome", "extensions", "file"]

def process_api_traces(apiTraces: List, site: str):
    siteName = site.split('/')[-1]
    total_records = len(apiTraces)
    th = 0
    for trace in apiTraces:
        try:
            record = json.loads(trace)
            if record["top_level_url"] == siteName:
                if 'scriptURL' in record:
                    script_url = record["scriptURL"]
                    if any(script_url.startswith(invalid) for invalid in INVALID_URLS):
                        th += 1
                    if th == total_records:
                        try:
                            print(siteName)
                            shutil.rmtree(site)
                        except Exception as e:
                                print("while removing the errori site this happened: " ,e)

        except Exception as e:
            print(e, siteName)
                

def main():
    chunks = utilities.get_directories_in_a_directory(Path(Path.cwd(), 'server/output'))
    
    for chunk in chunks:
        print(chunk)
        sites = utilities.get_directories_in_a_directory(chunk)
        print("Raw number of files: ", len(sites))
        for site in sites:
            site_name = site.split('/')[-1]
            site_content = utilities.get_files_in_a_directory(site)
            for file in site_content:
                if file.split('/')[-1] == 'apiTraces.json':
                    api_traces = utilities.read_file_splitlines(file)
                    process_api_traces(api_traces, site)
        # after process                
        sites = utilities.get_directories_in_a_directory(chunk)
        print("Final number of files: ", len(sites))
if __name__ == "__main__":
    main()