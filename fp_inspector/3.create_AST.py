import os
import esprima
import json
from multiprocessing import Manager
import itertools
from multiprocessing import Pool as ThreadPool
import multiprocessing
from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities 
import argparse


def  create_ast(raw_js_dir, ast_write_directory, cpu_to_relax):

    all_js_files_set = {file.name for file in raw_js_dir.iterdir()}
    processed_ast_set = {file.name.split('.json')[0] for file in ast_write_directory.iterdir()}

    log_file_path = Path(Path.cwd(), "obfuscation/data/logs/ast_beautifytools_construction.log")
    with log_file_path.open('r') as log_file:
        for line in log_file:
            # Assuming the first word in each line of the log is the file name
            processed_ast_set.add(line.strip().split(' ')[0])
    
    unprocessed_filenames = all_js_files_set - processed_ast_set
    unprocessed_js_files_list = [raw_js_dir / fname for fname in unprocessed_filenames]

    print(len(unprocessed_js_files_list))
    # pool = ThreadPool(processes=multiprocessing.cpu_count() - cpu_to_relax)
    pool = ThreadPool(processes=35)
    args = ((file, ast_write_directory, log_file_path) for file in unprocessed_js_files_list)
    results = pool.starmap(get_ast, args)
    pool.close()
    pool.join()


def get_ast(script_in_dir, ast_write_addr, log_file_path):
    #script_text = ldb.get(bytes(script_hash, 'utf-8')).decode()
    with open(script_in_dir, 'r', encoding='ISO-8859-1') as file:
        script_text = file.read()
    # script_text = utilities.read_dill_compressed(script_in_dir)
    script_hash = script_in_dir.name
    try:
        if 'ï»¿' in script_text:
            script_text = script_text.replace('ï»¿', '')
        print('Processing: ', script_hash)
        #script_text = utilities.read_full_file(script_text)
        # we capture the parsed scripts and use them in place of the packed versions when building ASTs
        ast = esprima.parseScript(script_text, options={'tolerant': True}).toDict()
        utilities.write_dill_compressed(Path(ast_write_addr, script_hash + '.json'), ast)
    except Exception as ex:
        print("Error while creating AST", str(ex))
        utilities.append_file(log_file_path, script_hash + ' Failed')
        pass


def main(js_type):
    
    base_dir = Path.cwd() / 'obfuscation/data'
    
    js_dir = base_dir / 'obfuscated_js' / js_type
    ast_write_directory = base_dir / 'AST' / js_type
    
    cpu_to_relax = 50
    if not os.path.exists(ast_write_directory):
        os.makedirs(ast_write_directory)
    create_ast(js_dir, ast_write_directory, cpu_to_relax)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process JavaScript files to generate ASTs.")
    parser.add_argument('js_type', type=str, help="The subdirectory within 'obfuscated_js' to use")
    
    args = parser.parse_args()
    main(args.js_type)