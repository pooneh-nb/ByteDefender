from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
import shutil


chunks = utilities.get_directories_in_a_directory(Path.home().joinpath(Path.home().cwd(), 'server/output'))

for chunk in chunks:
    sites = utilities.get_directories_in_a_directory(chunk)
    for site in sites:
        site_content = utilities.get_files_in_a_directory(site)
        if len(site_content) < 2:
            try:
                print(site.split('/')[-1])
                shutil.rmtree(site)
            except Exception as e:
                print(e)