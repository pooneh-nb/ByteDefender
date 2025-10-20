"""
This process creates a database. It includes tokenized bytecodes(separated with comma) 
which are labeled as True (FP) and False (Non-FP)
"""
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
sys.path.insert(1, Path.home().cwd().as_posix())
proj_path = Path.home().cwd()

import utilities
import logging

logging_path = Path.home().joinpath(proj_path, 'ML_Model/longformer/dataprocessing.log')
logging.basicConfig(filename=logging_path, filemode='a', 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    level=logging.ERROR)


js_type = ["raw_scripts", "ggl_obfuscated_scripts", "js_obfuscated_scripts"]
db_path = utilities.read_json(Path(Path.cwd(), f'create_DataBase/DB/sliced_dataset/script_level_DB/{js_type[2]}/DB.json'))

bytecodes = []
labels = []
for site, site_info in db_path.items():
    for script_id, script_info in site_info.items():
        bytecode_list = ' ,'.join(script_info["Bytecode"]).split(',')
        bytecode_sequences = [word.strip() for word in bytecode_list if word.strip()]
        label = script_info["label"]
        bytecodes.append(bytecode_sequences)
        labels.append(label)

utilities.write_json(Path(Path.cwd(), f'create_DataBase/DB/sliced_dataset/script_level_DB/{js_type[2]}/bytecodes.json'), bytecodes)
utilities.write_json(Path(Path.cwd(), f'create_DataBase/DB/sliced_dataset/script_level_DB/{js_type[2]}/labels.json'), labels)

