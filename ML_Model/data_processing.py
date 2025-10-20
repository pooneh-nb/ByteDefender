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

logging_path = Path.home().joinpath(proj_path, 'ML_Model/dataprocessing.log')
logging.basicConfig(filename=logging_path, filemode='a', 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    level=logging.ERROR)


# dbs_path = Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB_batch2')
dbs_path = Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB_batch1')
dbs = utilities.get_files_in_a_directory(dbs_path)

label_mapping = {False: 0, True: 1}

csv_output_path = Path.home().joinpath(dbs_path, 'dataset/bytecode_data.csv')
# csv_output_path = Path.home().joinpath(dbs_path, 'dataset/FP_bytecode_data.csv')

columns = ['Site', 'ScriptID', 'FunctionName', 'ParameterCount', 'RegisterCount', 'FrameSize', 'Bytecode', 'Label']

JUMBLING_WORDS = ['############### START ####################', 'Script URL:', 'Script ID:', 
                  'Function name:', 'Bytecode:', '############### END ####################']

fp = 0
non = 0
for db_dir in dbs:
    db_name = Path(db_dir).stem
    print(db_dir)
    db = utilities.read_json(db_dir)
    for site, meta in tqdm(db.items()):
        for script_id, script_info in meta.items():
            for func_name, func_info in script_info["functions"].items():
                # concatenate bytecode item into a single string
                bytecode_list = ','.join(func_info["Bytecode"][1:]).split(',')
                bytecode_sequences = [word.strip() for word in bytecode_list if word.strip() and not any(jumbling in word for jumbling in JUMBLING_WORDS)]
                # Map boolean label to numeric
                label = label_mapping[func_info["FP"]]
                
                if bytecode_sequences != []:
                # if bytecode_sequences != [] and label == 1:
                    if label == 0:
                        non += 1
                    else:
                        fp += 1
                    try:
                        parameter = False
                        register = False
                        frame = False

                        parameter_start = bytecode_sequences[0].split(' ')[0]
                        if parameter_start == 'Parameter':
                            parameter = True
                            parameter_count = int(bytecode_sequences[0].split(' ')[-1])

                        register_start = bytecode_sequences[1].split(' ')[0]
                        if register_start == 'Register':
                            register = True
                            register_count = int(bytecode_sequences[1].split(' ')[-1])

                        frame_start = bytecode_sequences[2].split(' ')[0]
                        if frame_start == 'Frame':
                            frame = True
                            frame_size = int(bytecode_sequences[2].split(' ')[-1])

                        if parameter and register and frame:
                            raw_byte = bytecode_sequences[3:]
                            # Write directly to CSV
                            with open(csv_output_path, 'a', newline='') as f:
                                pd.DataFrame([(site, script_id, func_name, parameter_count, 
                                                register_count, frame_size, raw_byte, label)], 
                                                columns=columns).to_csv(f, header=False, index=False)
                    except Exception as e:
                        logging.error(f"Error occurred in 'data_provessing' : {db_name} : {site} : {script_id} : {str(e)}")
                       
print(fp, non)
