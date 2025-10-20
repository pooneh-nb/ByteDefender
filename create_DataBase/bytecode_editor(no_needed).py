from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
from tqdm import tqdm


def process_browser_logs(site, output_file_path, start, end):
    
    try:
        files = utilities.get_files_in_a_directory(site)
        for file in files:
            if file.split('/')[-1] == 'browser_logs.bin':
                with open(file, 'r') as input_file, open(output_file_path, 'w') as output_file:
                    # Initialize variables to track the state
                    in_batch = False
                    current_batch = []
                    for line in input_file:
                        if line.startswith(start) and not in_batch:
                            in_batch = True
                            current_batch = [line]  # Start a new batch
                            continue
                        elif in_batch:
                            if line.startswith(start):
                                continue
                            if line.startswith(end):
                                if len(line) > len(end)+2:
                                    line = end+"\n"
                                current_batch.append(line) 
                                in_batch = False
                                output_file.writelines(current_batch)
                                current_batch = []  # Reset the current batch 
                                continue
                            current_batch.append(line) 
                        else:
                            continue
    except Exception as e:
        with open ('edit_byte.log', 'a') as file:
            file.writelines(e)

def main(CHUNK_NAME):

    raw_crawled_path = Path.home().joinpath(Path.home().cwd(),f"server/tranco_10k/{CHUNK_NAME}")
    raw_crawled_sites = utilities.get_directories_in_a_directory(raw_crawled_path)

    START = "############### START ####################"
    END = "############### END ####################"

    for site in tqdm(raw_crawled_sites):
        output_file_path = Path.home().joinpath(site, 'corrected_browser_logs.bin')
        process_browser_logs(site, output_file_path, START, END)