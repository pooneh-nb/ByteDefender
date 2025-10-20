from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
import csv
from tqdm import tqdm


sites = utilities.get_files_in_a_directory(Path.home().joinpath(Path.home().cwd(), 'sampled_sites'))
unify = ['website']

for site_list in sites:
    count = 10000
    site_list_name = site_list.split('/')[-1]
    print(site_list_name)
    if site_list.split('.')[-1] == 'csv':
        with open(site_list, 'r') as csv_file:
            file = csv.reader(csv_file, delimiter=',')
            for web in tqdm(file):
                if web[0] == ['website']:
                    continue
                if web[0] not in unify:
                    unify.append(web[0])
                    
print(len(unify))
with open(Path.home().joinpath(Path.home().cwd(), 'sampled_sites', 'unify.csv'), 'w', newline='\n') as unify_file:
    writer = csv.writer(unify_file)
    writer.writerow(unify) # Use writerow for single
            