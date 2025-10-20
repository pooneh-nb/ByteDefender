from pathlib import Path
import pandas as pd
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
from tqdm import tqdm

crawled_tranco_chunks = ['chunk_1.csv', 'chunk_2.csv', 'chunk_3.csv', 'chunk_4.csv', 'chunk_5.csv',
                        'chunk_6.csv', 'chunk_7.csv', 'chunk_8.csv', 'chunk_9.csv', 'chunk_10.csv',
                        'chunk_11.csv', 'chunk_12.csv', 'chunk_13.csv', 'chunk_14.csv', 'chunk_15.csv',
                        'chunk_16.csv', 'chunk_17.csv', 'chunk_18.csv', 'chunk_19.csv']

crawled_sampled_chunks = ['chunk_1.csv', 'chunk_2.csv', 'chunk_3.csv', 'chunk_4.csv', 'chunk_5.csv',
                          'chunk_6.csv', 'chunk_7.csv', 'chunk_8.csv', 'chunk_9.csv', 'chunk_10.csv',
                          'chunk_11.csv', 'chunk_12.csv', 'chunk_13.csv', 'chunk_14.csv', 'chunk_15.csv',
                          'chunk_16.csv', 'chunk_17.csv', 'chunk_18.csv', 'chunk_19.csv', 'chunk_20.csv', 
                          'chunk_21.csv', 'chunk_22.csv', 'chunk_23.csv', 'chunk_24.csv', 'chunk_25.csv', 
                          'chunk_26.csv', 'chunk_27.csv', 'chunk_28.csv', 'chunk_29.csv', 'chunk_30.csv', 
                          'chunk_31.csv', 'chunk_32.csv', 'chunk_33.csv', 'chunk_34.csv', 'chunk_35.csv', 
                          'chunk_36.csv', 'chunk_37.csv', 'chunk_38.csv']

tranco_path = Path.home().joinpath(Path.home().cwd(), 'sites/tranco')
tranco_chunks = utilities.get_files_in_a_directory(tranco_path)

sampled_path = Path.home().joinpath(Path.home().cwd(), 'sites/sampled_sites/chunks')
sampled_chunks = utilities.get_files_in_a_directory(sampled_path)

visited_sites = set()

for chunk in tqdm(tranco_chunks):
    chunk_name = chunk.split('/')[-1]
    if chunk_name in crawled_tranco_chunks:
        print(chunk_name)
        chunk_content = utilities.read_file_newline_stripped(chunk)
        for site in chunk_content:
            if site != 'website':
                visited_sites.add(site)

print(len(visited_sites))

for chunk in tqdm(sampled_chunks):
    chunk_name = chunk.split('/')[-1]
    if chunk_name in crawled_sampled_chunks:
        print(chunk_name)
        chunk_content = utilities.read_file_newline_stripped(chunk)
        for site in chunk_content:
            if site != 'website':
                visited_sites.add(site)
    print(len(visited_sites))

print(len(visited_sites))
utilities.write_json(Path.home().joinpath(Path.home().cwd(), 'sites/visited_sites.json'), list(visited_sites))

