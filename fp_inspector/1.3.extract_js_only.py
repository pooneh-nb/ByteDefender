from pathlib import Path
import logging
import json
from requests.exceptions import Timeout, ConnectionError, RequestException
import requests
from multiprocessing import Pool
import re
import os
import shutil
import hashlib
import pandas as pd
from tqdm import tqdm
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
proj_path = Path.cwd()
import utilities
from tqdm import tqdm

logging_path = Path(Path.cwd(), 'fp_inspector/extract_js_only.log')
logging.basicConfig(filename=logging_path, filemode='a', 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    level=logging.ERROR)

# dataset_path = Path(Path.cwd(), 'create_DataBase/DB/script_level/dataset/sticked_functions/filtered/concated_script_level_dataset.csv')
# # dataset_path = Path(Path.cwd(), 'create_DataBase/DB/script_level/dataset/sticked_functions/filtered/test_dataset.csv')
# with open(dataset_path) as f:
#     df = pd.read_csv(f)

def generate_unique_filename(url):
    """Generate a unique filename for the URL using MD5 hash."""
    hash_object = hashlib.md5(url.encode())
    return hash_object.hexdigest()

def download_js(args):
    site, url, label, js_source_path = args
    # Send a GET request to the URL
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            
            content_type = response.headers.get('Content-Type', '').lower()
            if 'javascript' in content_type or 'ecmascript' in content_type:
                unique_filename = generate_unique_filename(url)
                raw_js_path = Path(js_source_path, f"{unique_filename}_{label}.js")
                with open(raw_js_path, 'wb') as file:
                    file.write(response.content)
                return
            else:
                # print(f"HTTP error: {response.status_code}", site, url)
                logging.error(f"download_js: Content-Type mismatch in {site} : {url} : {content_type}")
                return

        except (Timeout, ConnectionError) as err:
            if attempt < max_retries:
                logging.info("Retrying...")
                continue
            else:
                logging.error(f"download_js: Error in {site} : {url} : timeout")
                return

        except RequestException as err:
            logging.error(f"download_js: Error in {site} : {url} : {str(err)}")
            break
    
    logging.error(f"download_js: Failed to download after {max_retries} attempts for URL: {site} : {url} ")
    return

def fetch_and_clean_log(log_file_path, stats, chunk_site):

    # Define regex patterns for each log type    
    patterns = [
        r"Error in (\S+) : (\S+) : \d+ \w+ Error: \w+ for url: (\S+)",
        r"Failed to download after \d+ attempts for URL: (\S+) : (\S+)",
        r"Content-Type mismatch in (\S+) : (\S+) :",
        r"Error in (\S+) : (\S+) : timeout"
    ]

    temp_file_path = Path(log_file_path.parent,'extract_js_only.tmp')
    with open(log_file_path, "r") as log_file, open(temp_file_path, "w") as temp_file:
        for line in tqdm(log_file, unit=" lines"):
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        site = match.group(1)
                        url = match.group(2)
                        print(url)
                        unique_filename = generate_unique_filename(url)
                        label = stats[site][url]['label']
                        js_source_path = Path(Path.cwd(), 'fp_inspector/data/JS', site)
                        js_source_path.mkdir(exist_ok=True)
                        raw_js_path = Path(js_source_path, f"{unique_filename}_{label}.js")
                        try:
                            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
                            response = requests.get(url, headers=headers, timeout=10)  # Sending request to the URL
                            if response.status_code == 200:
                                print(f"Successfully downloaded {url}")    
                                with open(raw_js_path, 'wb') as file:
                                    file.write(response.content)
                                continue  # Skip writing this line back if successful
                        except requests.RequestException as e:
                            temp_file.write(line)
                            print(f"Failed to download {url}: {e}")
    # Replace the old file with the new one
    os.replace(temp_file_path, log_file_path)

def main():

    # 1. collect js sources from extract_stats.json
    stats_path = Path(Path.cwd(), "fp_inspector/data/extract_stats.json")
    with open(stats_path, 'r', encoding='utf-8') as file:
        stats = json.load(file)

    # 2. revist sites with error
    log_file_path = Path(Path.cwd(), 'fp_inspector/extract_js_only.log')
    chunk_site = utilities.read_json(Path(Path.cwd(), 'fp_inspector/data/chunk_site.json'))
    fetch_and_clean_log(log_file_path, stats, chunk_site)

if __name__ == "__main__":
    main()