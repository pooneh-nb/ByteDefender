import json
import queue
import os
from multiprocessing import Manager
import itertools
from multiprocessing import Pool as ThreadPool
import multiprocessing
from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities 
import argparse


ALL = 'ALL'
NO_NAMES = 'NO_NAMES'
KEYWORD = 'KEYWORD'


# generate static features from ASTs. first hierarchically traverse the ASTs and divide them into pairs of parent and
# child node. Parents represent the context (e.g., for loops, try statements, or if conditions), and children represent
# the function inside that context (e.g., createElement, toDateURL, and measureText)[parent:child]

def new_walk_reserved_words(all_data_addr, features_directory, reserved_words, log_file_path):
    try:
        node_queue = queue.Queue()
        all_data = utilities.read_dill_compressed(all_data_addr)
        node_queue.put(('', all_data))

        node_list = []
        context_list = []
        text_list = []
        raw_features = set()
        first_run = True

        while not node_queue.empty():
            parent, data = node_queue.get()

            for key, value in data.items():
                if isinstance(value, dict):
                    node_list.append(value)
                elif isinstance(value, list) or isinstance(value, tuple):
                    if len(value) != 0:
                        for v in value:
                            node_list.append(v)
                else:
                    if key == 'type':
                        context_list.append(value)
                    else:
                        value_to_add = ''.join(
                            str(value).replace('\"', '').replace('\'', '').strip().replace(',', '').replace('\t',
                                                                                                            '').replace(
                                '\n', '').replace('\r', '').splitlines()).strip()
                        if value_to_add in reserved_words:  # naively parsing parent:child pairs for the entire ast of every script would result in a large set of features
                            text_list.append(value_to_add)

            for context in context_list:
                for node in node_list:
                    if node is not None:
                        node_queue.put((context, node))

            if first_run is False:
                for text in text_list:
                    raw_features.add(text)

            for context in context_list:
                for text in text_list:
                    raw_features.add(text)

            node_list = []
            context_list = []
            text_list = []
            first_run = False
        
        new_filename = all_data_addr.name.replace('.json', '.txt')
        output_path = Path(features_directory, new_filename)

        with open(output_path, 'w', encoding='utf-8') as out_file:  # Open the file in text mode
            for item in raw_features:
                out_file.write(item + '\n')  
        utilities.append_file(log_file_path, all_data_addr.name + ' Passed')
        # utilities.write_list_simple(os.path.join(features_directory, all_data_addr.split('/')[-1].replace('json', 'txt')), raw_features)
    except Exception as e:
        print('Error while processing: ', all_data_addr.name, str(e))
        utilities.append_file(log_file_path, all_data_addr.name + ' Failed')
    except EOFError as er:
        print('Error while processing: ', all_data_addr.name, str(er))
        utilities.append_file(log_file_path, all_data_addr.name + ' Failed')


def extract_features_multiprocess(ast_directory, features_directory, feature_type_to_extract, js_keywords_file,
                                  cpu_to_relax):
    ast_files = {file.name.split('.json')[0] for file in ast_directory.iterdir()}
    keywords_list = [key_js.strip() for key_js in utilities.read_file_newline_stripped(js_keywords_file)]
    feature_files = {file.name.split('.txt')[0] for file in features_directory.iterdir()}
    
    log_file_path = Path(Path.cwd(), "obfuscation/data/logs/ast_beautifytools_parsing.log")
    with log_file_path.open('r') as log_file:
        for line in log_file:
            # Assuming the first word in each line of the log is the file name
            if line.strip().split(' ')[-1] == 'Failed':
                feature_files.add(line.strip().split(' ')[0].split('.json')[0])

    unprocessed_filenames = ast_files - feature_files
    unprocessed_js_files_list = [ast_directory / (fname + '.json') for fname in unprocessed_filenames]
    

    print(len(unprocessed_js_files_list), 'files to process')
    # pool = ThreadPool(processes=multiprocessing.cpu_count() - cpu_to_relax)
    pool = ThreadPool(processes=50)
    try:
        args = ((file, features_directory, keywords_list, log_file_path) for file in unprocessed_js_files_list)
        results = pool.starmap(new_walk_reserved_words, args)
    except Exception as e:
        print('Exception in main thread: ', str(e))

    pool.close()
    pool.join()

    return


def main(js_type):
    base_dir = Path.cwd() / 'obfuscation/data'
    
    js_dir = base_dir / 'obfuscated_js' / js_type
    ast_directory = base_dir / 'AST' / js_type
    features_directory = base_dir / 'api_features' / js_type

    # ast_directory = Path(Path.cwd(), 'fp_inspector/data/AST')
    # features_directory = Path(Path.cwd(), 'fp_inspector/data/api_features')
    
    feature_type_to_extract = KEYWORD
    js_keywords_file = Path(Path.cwd(), 'fp_inspector/data/cleaned_apis_unique.txt')
    cpu_to_relax = 10

    if not os.path.exists(features_directory):
        os.makedirs(features_directory)

    extract_features_multiprocess(ast_directory, features_directory, feature_type_to_extract, js_keywords_file, cpu_to_relax)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process JavaScript files to generate ASTs.")
    parser.add_argument('js_type', type=str, help="The subdirectory within 'obfuscated_js' to use")
    
    args = parser.parse_args()
    main(args.js_type)
