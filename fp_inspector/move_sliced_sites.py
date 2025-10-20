import os
import sys
import ast
import json
import shutil
import logging
import itertools
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.insert(1, Path.home().cwd().as_posix())
proj_path = Path.home().cwd()
import utilities

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def copy_src_to_dst(src, dst):
    """
    Copy all files and directories from src to dst.
    If dst does not exist, it will be created.
    Existing files in dst with the same name will be overwritten.
    """
    try:
        if not os.path.exists(dst):
            os.makedirs(dst)
        
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)

            if os.path.isdir(s):
                copy_src_to_dst(s, d)
            else:
                shutil.copy2(s, d)
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")
        logging.error(f"Error copying {src} to {dst}: {e}")


def main():
    # Define the destination directory path
    chunk_site = utilities.read_json(Path.home().joinpath(Path.home().cwd(), 'fp_inspector/data/chunk_site.json'))
    db_path = Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/sliced_dataset/single_labeled_sliced_db.csv')
    db = pd.read_csv(db_path, header=None)
    logging.info(f"Database shape: {db.shape}")
    print(db.shape)
    
    counter = 0
    for idx, row in tqdm(db.iterrows()):
        counter += 1
        if idx == 0:
            continue
        # print(idx)
        site = row[0]
        script_id = row[1]
        batch = chunk_site[site][0]
        chunk = chunk_site[site][1]
        site_path = Path(Path.home().cwd(), 'server', batch, chunk, site)
        if not os.path.exists(site_path):
            logging.warning(f"Source path does not exist: {site_path}")
            print(f"Source path does not exist: {site_path}")
        destination_dir = Path(Path.cwd(), 'fp_inspector/data/new_sliced_db', site)
        # print(destination_dir)

        if not os.path.exists(destination_dir):
            print(f"Copying {site_path} to {destination_dir}")
            logging.info(f"Copying {site_path} to {destination_dir}")
            copy_src_to_dst(site_path, destination_dir)
    # print(counter)
                

if __name__ == "__main__":
    main()