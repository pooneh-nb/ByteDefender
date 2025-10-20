from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
proj_path = Path.home().cwd()
import utilities 


dic = {}

crawled_path = utilities.get_directories_in_a_directory(Path(Path.cwd(), 'server/crawled'))

counter = 1
for chunk in crawled_path:
    chunk_name = Path(chunk).stem
    sites = utilities.get_directories_in_a_directory(chunk)
    for site in sites:
        site_name = Path(site).name
        dic[site_name] = chunk_name
    # counter += 1

utilities.write_json(Path.home().joinpath(Path.cwd(), 'fp_inspector/data/chunk_site.json'), dic)

