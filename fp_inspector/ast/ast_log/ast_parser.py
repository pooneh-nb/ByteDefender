# for each ast extract features => parse asts
import queue
import os
from multiprocessing import Manager
import itertools
from multiprocessing import Pool as ThreadPool
from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
proj_path = Path.home().cwd()
import utilities
import logging
from tqdm import tqdm

logging_path = Path.home().joinpath(proj_path, 'fp_inspector/ast_parser.log')
logging.basicConfig(filename=logging_path, filemode='a', 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    level=logging.ERROR)

ALL = 'ALL'
NO_NAMES = 'NO_NAMES'
KEYWORD = 'KEYWORD'


# generate static features from ASTs. first hierarchically traverse the ASTs and divide them into pairs of parent and
# child node. Parents represent the context (e.g., for loops, try statements, or if conditions), and children represent
# the function inside that context (e.g., createElement, toDateURL, and measureText)[parent:child]


def new_walk_reserved_words(ast_dir, js_keywords, features_dir_out, chunk, site_name, script_id):
    try:
        node_queue = queue.Queue()
        ast = utilities.read_dill_compressed(ast_dir)
        node_queue.put(('', ast))

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
                        if value_to_add in js_keywords:  # naively parsing parent:child pairs for the entire ast of every script would result in a large set of features
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

        features_file_name = Path(ast_dir).stem + '.txt'
        output_file_path = Path.home().joinpath(features_dir_out, features_file_name)
        with open(output_file_path, 'w') as file:
            for feature in raw_features:
                file.write(f"{feature}\n")
    except Exception as e:
        logging.error(f"Error occurred in 'new_walk_reserved_words' : {chunk} : {site_name} : {script_id} : {str(e)}")
   

def moderate_multiprocessing(site, js_keywords, chunk):
    try:
        site_name = site.split('/')[-1]
        print(site_name)
        ast_path = Path.home().joinpath(site, 'ast')
        
        if os.path.exists(ast_path):
            asts = utilities.get_files_in_a_directory(ast_path)
            if len(asts) > 0:
                feature_writing_path = Path.home().joinpath(site, 'api_features')
                feature_writing_path.mkdir(exist_ok=True)
                for ast in asts:
                    script_id = Path(ast).stem
                    new_walk_reserved_words(ast, js_keywords, feature_writing_path, chunk, site_name, script_id)
                   
    except Exception as e:
        logging.error(f"Error occurred in 'moderate_multiprocessing' : {chunk} : {site_name} : {str(e)}")

def main():
    js_keywords = utilities.read_file_newline_stripped(Path.home().joinpath(proj_path, 'fp_inspector/cleaned_apis.unique.txt'))
    chunks = utilities.get_directories_in_a_directory(Path.home().joinpath(proj_path, 'server/crawled'))
    for chunk in tqdm(chunks):
        chunk_name = chunk.split('/')[-1]
        print(chunk_name)
        sites = utilities.get_directories_in_a_directory(chunk)
        pool = ThreadPool(processes=4)
        pool.starmap(moderate_multiprocessing, zip(sites, 
                                                   itertools.repeat(js_keywords), 
                                                   itertools.repeat(chunk_name)))
        pool.close()
        pool.join() 


if __name__ == '__main__':
    main()

# base_directory = '/media/umar/Elements/working_directory/processed_scripts'
# # # For ALL : START
# # result_directory = '/media/umar/Elements/working_directory/features_per_script'
# # js_keywords = '/media/umar/Elements/working_directory/JS_ALL_KEYWORDS.txt'
# # extract_features(base_directory, result_directory, js_keywords, ALL)
# # # For ALL : END

# # # For NO_NAMES : START
# # result_directory = '/media/umar/Elements/working_directory/features_per_script_noname'
# # js_keywords = '/media/umar/Elements/working_directory/JS_ALL_KEYWORDS.txt'
# # extract_features(base_directory, result_directory, js_keywords, NO_NAMES)
# # # For NO_NAMES : START

# # For KEYWORD : START
# result_directory = '/media/umar/Elements/working_directory/features_per_script_keyword'
# js_keywords = '/media/umar/Elements/working_directory/JS_ALL_KEYWORDS.txt'
# extract_features(base_directory, result_directory, js_keywords, KEYWORD)
# # For KEYWORD : END