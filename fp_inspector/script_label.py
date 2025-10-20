import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd

sys.path.insert(1, Path.home().cwd().as_posix())
proj_path = Path.home().cwd()
import utilities

db_path = Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/sliced_dataset/single_labeled_sliced_db.csv')
db = pd.read_csv(db_path, header=None)

script_label = {}
for idx, row in tqdm(db.iterrows()):
    if idx == 0:
        continue
    site_name = row[0]
    script_id = row[1]
    label = row[7]

    if site_name not in script_label:
        script_label[site_name] = {}
    if script_id not in script_label[site_name]:
        script_label[site_name][script_id] = label
    script_label[site_name][script_id] = label

out_path = Path(Path.cwd(), 'fp_inspector/data/script_label.json')
utilities.write_json(out_path, script_label)
     