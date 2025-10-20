from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
proj_path = Path.home().cwd()
import utilities 
import os
import shutil


def collect_all_js_files_in_a_directory(js_files, raw_js_dir):
    copied_files = utilities.get_files_in_a_directory(raw_js_dir)
    for js_file in js_files:
        js_name = js_file.split('/')[-1]
        site = js_file.split('/')[-2]
        dest_file = Path(raw_js_dir, site + "_" + js_name)
        try:
            fileSize = round(os.path.getsize(js_file) / 1024, 2)  # File size in KB
            if str(dest_file) not in copied_files:
                if fileSize == 0:  # Check if file is empty
                    print(f"Empty file not copied: {site}_{js_name}")
                else:
                    shutil.copy(js_file, dest_file)
                    print(f"Copied: {site}_{js_name}")
            else:
                if fileSize == 0:
                    os.remove(dest_file)
                    print(f"Removed empty file: {dest_file}")
        except Exception as e:
            print(f"Error processing file {js_file}: {e}")


def main():
    sites = utilities.get_directories_in_a_directory(Path(Path.cwd(), 'fp_inspector/data/JS'))
    js_files = (file for site in sites for file in utilities.get_files_in_a_directory(site))
    raw_js_dir = Path(Path.cwd(), 'fp_inspector/data/raw_js')
    collect_all_js_files_in_a_directory(js_files, raw_js_dir)