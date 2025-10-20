import pandas as pd
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
proj_path = Path.cwd()
import utilities

import shutil
from multiprocessing import Pool

def process_site(site, chunk_site):
    chunk = chunk_site[site]
    site_path = Path(Path.cwd(), 'server/crawled', chunk, site)
    js_source_path = Path(site_path, 'js')

    # Remove the 'js' directory if it exists
    if js_source_path.exists() and js_source_path.is_dir():
        shutil.rmtree(js_source_path)
    return site  # Returning site for any follow-up or logging

def main():
    chunk_site = utilities.read_json(Path(Path.cwd(), 'fp_inspector/data/chunk_site.json'))
    db_path = Path(Path.cwd(), 'create_DataBase/DB/script_level/dataset/sticked_functions/filtered/concated_script_level_dataset.csv')
    # db_path = Path(Path.cwd(), 'create_DataBase/DB/script_level/dataset/sticked_functions/filtered/test_dataset.csv')
    db = pd.read_csv(db_path, header=None, skiprows=1)
    print(db.shape)

    # Get unique sites directly from the DataFrame
    unique_sites = db.iloc[:, 0].unique()  # Assuming 'site' is in the first column

    # Create a pool of worker processes
    with Pool(processes=20) as pool:  # Adjust number of processes based on your CPU
        results = list(tqdm(pool.starmap(process_site, [(site, chunk_site) for site in unique_sites]), total=len(unique_sites)))

    print("Processed sites:", results)

if __name__ == '__main__':
    main()

# chunk_site = utilities.read_json(Path(Path.cwd(), 'fp_inspector/data/chunk_site.json'))

# db_path = Path(Path.cwd(), 'create_DataBase/DB/script_level/dataset/sticked_functions/filtered/concated_script_level_dataset.csv')
# # db_path = Path(Path.cwd(), 'create_DataBase/DB/script_level/dataset/sticked_functions/filtered/test_dataset.csv')

# db = pd.read_csv(db_path, header=None, skiprows=1)
# print(db.shape)

# sites = set()

# for idx, row in tqdm(db.iterrows(), total=db.shape[0], desc="Processing rows"):
    
#     site = row[0]
#     if site not in sites:
#         sites.add(site)
#         script_URL = row[2]
#         chunk = chunk_site[site]

#         site_path = Path(Path.cwd(), 'server/crawled', chunk, site)
        
#         js_source_path = Path(site_path, 'js')

#         # Remove the 'js' directory if it exists
#         if js_source_path.exists() and js_source_path.is_dir():
#             shutil.rmtree(js_source_path)
