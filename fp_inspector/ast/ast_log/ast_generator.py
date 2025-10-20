# create a ast directory and for each script in js directory generate an ast and buffer them in the ast directory
import logging
import os
from pathlib import Path
import sys
from tqdm import tqdm
import esprima
import itertools
from multiprocessing import Pool as ThreadPool


sys.path.insert(1, Path.home().cwd().as_posix())
proj_path = Path.home().cwd()
import utilities

logging_path = Path.home().joinpath(proj_path, 'fp_inspector/ast_generator.log')
logging.basicConfig(filename=logging_path, filemode='a', 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    level=logging.ERROR)

def moderate_multiprocessing(site, chunk):
    site_name = site.split('/')[-1]
    # print(site_name)
    try:
        js_source_path = Path.home().joinpath(site, 'js')
        if os.path.exists(js_source_path):
            script_sources = utilities.get_files_in_a_directory(js_source_path)
            if len(script_sources) > 0:
                ast_writing_path = Path.home().joinpath(site, 'ast')
                ast_writing_path.mkdir(exist_ok=True)
                get_ast(script_sources, ast_writing_path, site, chunk)
    except Exception as e:
        logging.error(f"Error occurred in 'moderate_multiprocessing' : {chunk} : {site_name} : {str(e)}")
        


def get_ast(script_sources, ast_writing_path, site, chunk):
    site_name = site.split('/')[-1]
    unsuccessful_ast_log_path = Path.home().joinpath(Path.home().cwd(), 'fp_inspector/ast_log/unsucessful_js.txt') 
    for script in script_sources:
               
        script_id = Path(script).stem
        script_text = utilities.read_full_file(script)
        try:
            if 'ï»¿' in script_text:
                script_text = script_text.replace('ï»¿', '')
            ast = esprima.parseScript(script_text, options={'tolerant': True}).toDict()
            utilities.write_dill_compressed(Path.home().joinpath(ast_writing_path, f"{script_id}.json"), ast)

        except Exception as e:
            logging.error(f"Error occurred in 'get_ast' : {chunk} : {site_name} : {script_id} : {str(e)}")
            with open(unsuccessful_ast_log_path, 'a') as log_file:
                log_file.write(f"{script}\n")

def retry_failed_ast_generation(log_path):
    unsuccessful_scripts = utilities.read_file_newline_stripped(log_path)
    print(len(unsuccessful_scripts))
    still_failing_scripts = []

    for script in unsuccessful_scripts:
        try:
            site = script.split('/')[-3]
            script_id = Path(script).stem
            chunk = script.split('/')[-4]
            ast_writing_path = Path.home().joinpath(Path(script.split('js')[0]), 'ast')
            ast_writing_path.mkdir(exist_ok=True)
            script_text = utilities.read_full_file(script)
            if 'ï»¿' in script_text:
                script_text = script_text.replace('ï»¿', '')
            ast = esprima.parseScript(script_text, options={'tolerant': True}).toDict()
            utilities.write_dill_compressed(ast_writing_path.joinpath(f"{script_id}.json"), ast)
            # Optionally, remove successful entries from the log or mark them as processed
        except Exception as e:
            logging.error(f"Retry error: {chunk}:{site}:{script_id} - {str(e)}")
            still_failing_scripts.append(script)

    # Rewrite the log file with only the scripts that failed again
    print(len(still_failing_scripts))
    with open(log_path, 'w') as file:
        for script in still_failing_scripts:
            file.write(f"{script}\n") 

def main():
    chunks = utilities.get_directories_in_a_directory(Path.home().joinpath(proj_path, 'server/crawled'))
    # chunks.append(utilities.get_directories_in_a_directory(Path.home().joinpath(proj_path, 'server/batch1_crawled')))
    
    for chunk in tqdm(chunks):
        chunk_name = chunk.split('/')[-1]
        sites = utilities.get_directories_in_a_directory(chunk)
        pool = ThreadPool(processes=4)
        pool.starmap(moderate_multiprocessing, zip(sites, itertools.repeat(chunk_name)))
        pool.close()
        pool.join() 

    # unsuccessful_ast_log_path = Path.home().joinpath(Path.home().cwd(), 'fp_inspector/ast_log/unsucessful_js.txt') 
    # retry_failed_ast_generation(unsuccessful_ast_log_path)
            
if __name__ == "__main__":
    main()