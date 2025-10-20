from pathlib import Path
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(1, Path.cwd().as_posix())
import utilities

def read_csvs(csv_path):
    all_dataframes = []  # List to store each DataFrame

    # List all zip files in the given directory
    files = utilities.get_files_in_a_directory(csv_path)

    print("concatenating in process ...")
    for filename in files:
        if filename.endswith(".csv"):
            print(filename)
            # Read the CSV file contained within the zip file
            with open(filename) as f:
                df = pd.read_csv(f)
                print(df.shape[0])
                all_dataframes.append(df)

    # Concatenate all dataframes into a single dataframe
    print("concatenating")
    full_dataframe = pd.concat(all_dataframes, ignore_index=True)
    return full_dataframe

# Usage
csv_path = Path(Path.cwd(), 'create_DataBase/DB/script_level/dataset')
combined_df = read_csvs(csv_path)
print(combined_df.shape[0])
print("saving in process ...")
combined_df.to_csv(Path(csv_path, 'script_level_dataset.csv'), index=False)
print("done!")