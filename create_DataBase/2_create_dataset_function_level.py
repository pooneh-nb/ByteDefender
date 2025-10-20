"""
This process creates a database. It includes tokenized bytecodes(separated with comma) 
which are labeled as True (FP) and False (Non-FP)
"""
import sys
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
sys.path.insert(1, Path.home().cwd().as_posix())
proj_path = Path.home().cwd()
import utilities
import logging
import multiprocessing as mp


label_mapping = {False: 0, True: 1}

columns = ['Site', 'ScriptID', 'Script_URL', 'FunctionName', 'ParameterCount', 'RegisterCount', 'FrameSize', 'Bytecode', 'Label']

JUMBLING_WORDS = ['############### START ####################', 'Script URL:', 'Script ID:', 
                  'Function name:', 'Bytecode:', '############### END ####################']

    

def bytecode_to_numbers(bytecode_list, vocab):
    """Convert bytecode list to numbers based on vocabulary."""
    return [vocab.setdefault(word.strip(), len(vocab) + 1) for word in bytecode_list if word.strip() and not any(jumbling in word for jumbling in JUMBLING_WORDS)]

def process_chunk(args):
    
    db_dir, vocab = args 
    db_name = Path(db_dir).stem    

    logging_path = Path(proj_path, f'create_DataBase/DB/function_level/dataset/logs/{db_name}_dataprocessing.log')
    logging.basicConfig(filename=logging_path, filemode='a', 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                        level=logging.ERROR)
    
    csv_output_path = Path('create_DataBase/DB/function_level', f'dataset/{db_name}_bytecode_data.csv')
    csv_output_path.parent.mkdir(parents=True, exist_ok=True) 
    data_rows = []

   
    if not csv_output_path.exists():
        print(db_dir)
        db = utilities.read_json(db_dir)
        for siteName, meta in tqdm(db.items()):
            for scriptURL, script_info in meta.items():
                for scriptId, script_meta in script_info.items():
                    for functionName, func_info in script_meta.items():
                        # concatenate bytecode item into a single string
                        bytecode_list = ','.join(func_info["Bytecode"]).split(',')
                        bytecode_sequences = [word.strip() for word in bytecode_list if word.strip() and not any(jumbling in word for jumbling in JUMBLING_WORDS)]
                        # Map boolean label to numeric
                        label = label_mapping[func_info["FP"]]
                        
                        if bytecode_sequences != []:
                            try:
                                parameter = False
                                register = False
                                frame = False

                                list_start = bytecode_sequences[0].split(' ')[0]
                                if list_start == 'Parameter':
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
                                        # raw_byte = bytecode_to_numbers(bytecode_sequences[3:], vocab)
                                        raw_byte = bytecode_sequences[3:]
                                        data_rows.append((siteName, scriptId, scriptURL, functionName, parameter_count, register_count, frame_size, raw_byte, label))
                                        # raw_byte = bytecode_sequences[3:]
                                        # Write directly to CSV
                                        # pd.DataFrame([(siteName, scriptId, scriptURL, functionName, parameter_count, register_count, frame_size, raw_byte, label)],
                                        #             columns=columns).to_csv(csv_output_path, mode='a', header=not csv_output_path.exists(), index=False)
                            except Exception as e:
                                logging.error(f"Error occurred in 'data_provessing' : {db_name} : {siteName} : {scriptId} : {str(e)}")

        if data_rows:
            pd.DataFrame(data_rows, columns=columns).to_csv(csv_output_path, index=False)   
        print(db_name, "is done!")
        data_rows = []
        db = {}

        vocab_output_path = Path(Path.cwd(), 'create_DataBase/DB/function_level/dataset/vocab.json')
        with open(vocab_output_path, 'w') as f:
            json.dump(vocab, f)
# print(fp, non)

def main():
    # vocab = {}
    dbs_path = Path(Path.cwd(), 'create_DataBase/DB/function_level')
    dbs = utilities.get_files_in_a_directory(dbs_path)

    vocab_output_path = Path(dbs_path, 'dataset/vocab.json')
    with open(vocab_output_path, 'r') as f:
        vocab = json.load(f)

    # Create a pool of processes to handle each JSON file

    # for db_file  in dbs:
    #     process_chunk(db_file, vocab)
    args = [(db_file, vocab) for db_file in dbs]
    with mp.Pool(processes=5) as pool:
        pool.map(process_chunk, args)


if __name__ == "__main__":
    main()