from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
proj_path = Path.home().cwd()
import utilities

dic = {}

base_path = Path.home().joinpath(Path.cwd(), 'server')
batch1_crawled = utilities.get_directories_in_a_directory(Path(base_path, 'batch1_crawled'))
batch2_crawled = utilities.get_directories_in_a_directory(Path(base_path, 'batch2_crawled'))
batch3_crawled = utilities.get_directories_in_a_directory(Path(base_path, 'batch3_crawled'))

batches = [batch1_crawled, batch2_crawled, batch3_crawled]

counter = 1
for batch in batches:
    for chunk in batch:
        chunk_name = Path(chunk).stem
        sites = utilities.get_directories_in_a_directory(chunk)
        for site in sites:
            site_name = Path(site).name
            dic[site_name] = [f"batch{counter}_crawled", chunk_name]
    counter += 1

utilities.write_json(Path.home().joinpath(Path.cwd(), 'fp_inspector/data/chunk_site.json'), dic)

