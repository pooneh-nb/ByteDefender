from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
from tqdm import tqdm

# Define the output file paths
output_path = Path.home().joinpath(Path.home().cwd(),"server/output")
sites = utilities.get_directories_in_a_directory(output_path)

start = "############### START ####################"
end = "############### END ####################"

# Initialize variables to track the state
in_batch = False
current_batch = []
for site in tqdm(sites):
    try:
        files = utilities.get_files_in_a_directory(site)
        for file in files:
            if file.split('/')[-1] == 'browser_logs.bin':
                output_file_path = Path.home().joinpath(site, 'corrected_browser_logs.bin')
                with open(file, 'r') as input_file, open(output_file_path, 'w') as output_file:
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
                            if line.startswith("http"):
                                line = 'Script URL: '+ line
                            current_batch.append(line) 
                        else:
                            continue
    except Exception as e:
        with open ('edit_byte.log', 'a') as file:
            file.writelines(e)