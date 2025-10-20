import os
import sys

import logging
import subprocess
from pathlib import Path

sys.path.insert(1, Path.home().cwd().as_posix())
proj_path = Path.home().cwd()
import utilities
from tqdm import tqdm

log_path = Path(Path.cwd(), 'fp_inspector/obfuscation/google_closure.log')
logging.basicConfig(filename= log_path, level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')


sites = utilities.get_directories_in_a_directory(Path(Path.home().cwd(), 'fp_inspector/data/new_sliced_db'))

count = 0
for site in tqdm(sites):
    print(site)
    obfuscator_path = Path.home().joinpath(site, 'obfuscator')
    obfuscator_path.mkdir(exist_ok=True)

    ggl_obfuscator_path = Path.home().joinpath(obfuscator_path, 'google_closure')
    ggl_obfuscator_path.mkdir(exist_ok=True)
    
    js_path = Path(site, 'js')
    if os.path.exists(js_path):
        scripts = utilities.get_files_in_a_directory(Path(site, 'js'))
        
        for script in scripts:
            script_name = Path(script).name
            output_obfuscated = Path(ggl_obfuscator_path, script_name)
            print(output_obfuscated)

            if not os.path.exists(output_obfuscated):
                command = ['npx', 'google-closure-compiler', '--js', script, '--js_output_file', output_obfuscated]

                # Execute the command
                result = subprocess.run(command, capture_output=True, text=True)

                # Check if the command was executed successfully
                if result.returncode == 0:
                    print("Done!")
                    logging.error(f"Done: {site} : {script_name}")    
                else:
                    print("Error!")
                    logging.error(f"Error: {site} : {script_name}: {result.stderr}")
    else:
        os.mkdir(js_path)
        pass



