# for each website check the apiTraces to see which scripts with what id has API traces (keep them in a set)
# for these script ids, check the scripts.json and extract the source
# create a directory and keep javascripts with id
import json
import logging
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
import itertools
import requests
import ast
from multiprocessing import Pool, cpu_count

sys.path.insert(1, Path.home().cwd().as_posix())
proj_path = Path.cwd()
import utilities


logging_path = Path(proj_path, 'fp_inspector/script_source_extractor.log')
logging.basicConfig(filename=logging_path, filemode='a', 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    level=logging.ERROR)


def download_js(url, path, script_id):
    # Send a GET request to the URL
    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '').lower()
            if 'html' in content_type:
                raw_js_path = Path(path, f"{script_id}.html")
                with open(raw_js_path, 'wb') as file:
                    file.write(response.content)
                return "hashtml"
            
            # Determine if the content is JavaScript
            elif 'javascript' in content_type or 'ecmascript' in content_type:
                raw_js_path = Path(path, f"{script_id}.js")
                with open(raw_js_path, 'wb') as file:
                    file.write(response.content)
                    return "hasJS"
            
        else:
            return f"HTTP error: {response.status_code}"

    except requests.exceptions.Timeout:
        logging.error(f"download_js: Error in {path} : {str(script_id)} : timeout")
        return "timeout"

    except Exception as e:
        logging.error(f"download_js: Error in {path} : {str(script_id)} : {str(e)}")
        return str(e)

def process_row(args):
    row, chunk_site = args
    site = row[0]
    script_Id = int(row[1])
    script_URL = row[2]
    if script_URL == "https://cdn.livechatinc.com/widget/static/js/1.26d297cc.chunk.js":
        pass
    label = row[4]
    chunk = chunk_site[site]
    site_path = Path(Path.cwd(), 'server/crawled', chunk, site)
    # scripts = utilities.read_file_splitlines(Path(site_path, 'scripts.json'))
    
    entry = {}
    if script_URL not in entry:
        entry[script_URL] = {'status': "", 'label': label}
    # if script_Id not in entry[script_URL]:
    #     entry[script_URL][script_Id] = {'status': "", 'label': label}
    try:
        js_source_path = Path(site_path, 'js')
        js_source_path.mkdir(exist_ok=True)

        # Download js
        js_status = download_js(script_URL, js_source_path, script_Id)
        
        if js_status == "hasJS":
            entry[script_URL]['status'] = "hasJS"
        else:
            entry[script_URL]['status'] = js_status

    except Exception as e:
        print(f"Error processing {script_Id} at {script_URL}: {e}")
        logging.error(f"process_row: Error in 'site_path' : {str(script_Id)} : {str(e)}")

    return site, entry



def main():
    chunk_site = utilities.read_json(Path(Path.cwd(), 'fp_inspector/data/chunk_site.json'))

    db_path = Path(Path.cwd(), 'create_DataBase/DB/script_level/dataset/sticked_functions/filtered/concated_script_level_dataset.csv')
    # db_path = Path(Path.cwd(), 'create_DataBase/DB/script_level/dataset/sticked_functions/filtered/test_dataset.csv')

    db = pd.read_csv(db_path, header=None)
    dic = {}
    print(db.shape)

    data = [(row, chunk_site) for row in db.iloc[1:].values]

    # Create a pool of workers
    with Pool(20) as pool:
        results = list(tqdm(pool.imap(process_row, data), total=len(data), desc="Processing rows"))
    # for idx, row in db.iterrows():
    #     process_row(row, c)

    # Combine results into a single dictionary
    for site, entry in results:
        if site not in dic:
            dic[site] = entry
        else:
            for script_URL in entry:
                if script_URL not in dic[site]:
                    dic[site][script_URL] = entry[script_URL]
                else:
                    dic[site][script_URL].update(entry[script_URL])

    utilities.write_json(Path(Path.cwd(), 'fp_inspector/data/extract_stats.json'), dic)
    # success = 0
    # for idx, row in tqdm(db.iterrows(), total=db.shape[0], desc="Processing rows"):
    # # for idx, row in db.iterrows():
    #     if idx == 0:
    #         continue
    #     site = row[0]
    #     script_Id = int(row[1])
    #     script_URL = row[2]
    #     label = row[4]
    #     chunk = chunk_site[site]

    #     site_path = Path(Path.cwd(), 'server/crawled', chunk, site)
    #     # print(site_path)
    #     scripts = utilities.read_file_splitlines(Path(site_path, 'scripts.json'))
    #     if site not in dic:
    #         dic[site] = {}
    #     if script_URL not in dic[site]:
    #         dic[site][script_URL] = {}
    #     if script_Id not in dic[site][script_URL]:
    #         dic[site][script_URL][script_Id] = {'hasJS': False, 'label': label}
    #     try:
    #         js_source_path = Path(site_path, 'js')
    #         js_source_path.mkdir(exist_ok=True)

    #         # js_source = get_script_sources(script_URL, script_Id, scripts, site_path)
    #         # download js
    #         js_source = download_js(script_URL, js_source_path, script_Id)
            
    #         if js_source:
    #             dic[site][script_URL][script_Id]['hasJS'] = True
    #             # buffer_source(js_source, js_source_path, script_Id)
    #     except Exception as e:
    #         print(f"main: Error in 'site_path' : {str(script_Id)} : {str(e)}")
    #         logging.error(f"main: Error in 'site_path' : {str(script_Id)} : {str(e)}")

    # utilities.write_json(Path.home().joinpath(Path.cwd(), 'fp_inspector/data/extraxt_stats.json'), dic)


if __name__ == "__main__":
    main()