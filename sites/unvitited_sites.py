from pathlib import Path
import pandas as pd
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
from tqdm import tqdm
import csv

crawled_tranco_chunks = ['chunk_1.csv', 'chunk_2.csv', 'chunk_3.csv', 'chunk_4.csv', 'chunk_5.csv',
                        'chunk_6.csv', 'chunk_7.csv', 'chunk_8.csv', 'chunk_9.csv', 'chunk_10.csv',
                        'chunk_11.csv', 'chunk_12.csv', 'chunk_13.csv', 'chunk_14.csv', 'chunk_15.csv',
                        'chunk_16.csv', 'chunk_17.csv', 'chunk_18.csv', 'chunk_19.csv']

sites_path = Path.home().joinpath(Path.home().cwd(), 'sites')
visited_sites = utilities.read_json(Path.home().joinpath(sites_path, 'visited_sites.json'))

tranco_path = Path.home().joinpath(sites_path, 'tranco')
tranco_chunks = utilities.get_files_in_a_directory(tranco_path)

sampled_path = Path.home().joinpath(sites_path, 'sampled_sites/chunks')
sampled_chunks = utilities.get_files_in_a_directory(sampled_path)

# tranco_1m_path = Path.home().joinpath(Path.home().cwd(), 'sites/tranco_1m.csv')
# tranco_1m = pd.read_csv(tranco_1m_path)

new_sites = ['website']
counter = 20

for chunk in tqdm(tranco_chunks):
    chunk_name = chunk.split('/')[-1]
    if chunk_name not in crawled_tranco_chunks:
        print(chunk_name)
        chunk_content = utilities.read_file_newline_stripped(chunk)
        for site in chunk_content:
            if site != 'website' and site not in visited_sites:
                new_sites.append(site)
                if len(new_sites) % 3000 == 0:
                    with open(Path.home().joinpath(sites_path, f'res_tranco/chunk_{counter}.csv'), 'w') as file:
                        writer = csv.writer(file)
                        for item in new_sites:
                            writer.writerow([item])
                    new_sites = ['website']
                    counter += 1


print(len(new_sites))



# utilities.write_json(Path.home().joinpath(Path.home().cwd(), 'sites/visited_sites.json'), list(visited_sites))

