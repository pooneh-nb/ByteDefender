from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
import shutil

chunks = utilities.get_directories_in_a_directory(
    Path.home().joinpath(Path.home().cwd(), 
    'server/crawled'))

crawled_sites = set()

for chunk in chunks:
    print(chunk)
    sites = utilities.get_directories_in_a_directory(chunk)
    for site in sites:
        siteName = site.split('/')[-1]
        if siteName not in crawled_sites:
            crawled_sites.add(siteName)
        else:
            # this means that we have already seen the site in another chunk
            shutil.rmtree(site)