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
import ast
from multiprocessing import Pool as ThreadPool

sys.path.insert(1, Path.home().cwd().as_posix())
proj_path = Path.home().cwd()
import utilities


logging_path = Path.home().joinpath(proj_path, 'fp_inspector/script_source_extractor.log')
logging.basicConfig(filename=logging_path, filemode='a', 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    level=logging.ERROR)


def get_script_sources(script_id, scripts, site_path):
    """Finds and returns the source of a script by its ID."""

    for script in scripts:
        try:
            # script_record = ast.literal_eval(script)
            script_record = json.loads(script)
            if script_record['scriptId'] == script_id:
                return script_record['scriptSource']
        except Exception as e:
            logging.error(f"Error in {site_path} : {str(script_id)} : {str(e)}")

def buffer_source(script_source, path, script_id):
    """Saves the script source to a file."""
    try:
        raw_js_path = Path.home().joinpath(path, f"{script_id}.js")
        utilities.write_content(raw_js_path, script_source)
    except Exception as e:
        logging.error(f"Error occurred in 'buffer_source' : {str(e)}")


def main():
    chunk_site = utilities.read_json(Path.home().joinpath(Path.home().cwd(), 'fp_inspector/data/chunk_site.json'))
    db_path = Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/sliced_dataset/single_labeled_sliced_db.csv')
    db = pd.read_csv(db_path, header=None)
    dic = {}
    print(db.shape)
    success = 0
    for idx, row in db.iterrows():
        print(idx)
        if idx == 0:
            continue
        site = row[0]
        script_id = row[1]
        batch = chunk_site[site][0]
        chunk = chunk_site[site][1]
        site_path = Path(Path.home().cwd(), 'server', batch, chunk, site)
        # print(site_path)
        scripts = utilities.read_file_splitlines(Path(site_path, 'scripts.json'))
        if site not in dic:
            dic[site] = {}
        if script_id not in dic[site]:
            dic[site][script_id] = False
        try:
            js_source_path = Path.home().joinpath(site_path, 'js')
            js_source_path.mkdir(exist_ok=True)

            js_source = get_script_sources(script_id, scripts, site_path)
            
            if js_source:
                dic[site][script_id] = True
                buffer_source(js_source, js_source_path, script_id)
        except Exception as e:
            print(f"Error in 'site_path' : {str(script_id)} : {str(e)}")
            logging.error(f"Error in 'site_path' : {str(script_id)} : {str(e)}")

    utilities.write_json(Path.home().joinpath(Path.cwd(), 'fp_inspector/data/extraxt_stats.json'), dic)


if __name__ == "__main__":
    main()