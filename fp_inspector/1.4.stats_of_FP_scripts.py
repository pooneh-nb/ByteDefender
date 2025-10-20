from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
proj_path = Path.home().cwd()
import utilities 
import os
import math
import csv


def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return f"{s} {size_name[i]}"

def main():

    js_dir = utilities.get_directories_in_a_directory(Path('fp_inspector/data/JS'))
    
    stats = [['site', 'scriptName', 'fileSize', 'label']]
    for site in js_dir:
        siteName = site.split('/')[-1]
        scripts = utilities.get_files_in_a_directory(site)
        for script in scripts:
            fileSize = round(os.path.getsize(script) / 1024, 2)
            # with open(script, 'r', encoding='ISO-8859-1') as file:
            #     script_text = file.read()
            
            # yep = 0
            # if "eval(" in script_text or "Function(" in script_text:
            #     yep += 1
            scriptName = script.split('/')[-1].split('_')[0]
            label = script.split('/')[-1].split('_')[-1].split('.')[0]
            stats.append([siteName, scriptName, fileSize, label])
    
    output = Path('fp_inspector/data/script_labels.csv')
    with open(output, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(stats)
            
if __name__ == "__main__":
    main()