import sys
import json
from pathlib import Path
sys.path.insert(1, Path.cwd().as_posix())
import utilities
import pandas as pd
import ast




def update_vocab_and_convert(bytes, vocab):
    mapped_bytes = []
    for byte in bytes:
        byte = str(byte)  # Convert each byte to string to handle uniformly in vocabulary
        if byte not in vocab:
            vocab[byte] = len(vocab) + 1  # Assign a new unique integer ID
        mapped_bytes.append(vocab[byte])
    return mapped_bytes

def process_scripts(df, vocab, dataset_name):
    # clean dataset
    print("clenaing dataset")
    df = df.dropna().drop_duplicates()

    separator_token = '<sep>'  # Define a unique separator token

    # Group by script identifier
    # df['Bytecode'] = df['Bytecode'].apply(ast.literal_eval)
    print("convert bytecode column to list")
    df.loc[:, 'Bytecode'] = df['Bytecode'].apply(ast.literal_eval)
    grouped = df.groupby(['Site', 'ScriptID', 'Script_URL'])

    # Concatenate bytecodes and handle labels
    print("Concatenate bytecodes and handle labels")
    result_df = grouped.apply(lambda group: {
        'Bytecode': sum(([str(row['ParameterCount']), str(row['RegisterCount']), str(row['FrameSize'])] + row['Bytecode'] + 
                         [separator_token] if not str(row['FunctionName']).startswith('anonymous_func') else [separator_token]
                            for index, row in group.iterrows()), [])[:-1],  # Concatenate and remove the last separator
        'Label': group['Label'].max()  # Script label is 1 if any function is 1
    }).reset_index()

    # Split the processed groups into separate columns
    result_df[['Bytecode', 'Label']] = pd.DataFrame(result_df[0].tolist(), index=result_df.index)
    result_df.drop(columns=0, inplace=True)  # Remove the temporary column

    # Map the entire bytecode sequence to integers
    print("Map the entire bytecode sequence to integers")
    result_df['Bytecode'] = result_df['Bytecode'].apply(lambda x: update_vocab_and_convert(x, vocab))

    # Remove the separator token from the beginning and end of the list if present
    separator_id = vocab[separator_token]
    result_df['Bytecode'] = result_df['Bytecode'].apply(
        lambda x: x[1:-1] if len(x) > 1 and (x[0] == separator_id and x[-1] == separator_id) \
            else x[1:] if len(x) > 0 and x[0] == separator_id \
            else x[:-1] if len(x) > 0 and x[-1] == separator_id \
            else x
    )


    # Save the result to a CSV file
    script_dataset_path = Path(Path.cwd(), f'create_DataBase/DB/script_level/dataset/sticked_functions/{dataset_name}')
    result_df.to_csv(script_dataset_path, index=False)
    print(f"Data saved to {script_dataset_path}")

    vocab_path = Path(Path.cwd(), 'create_DataBase/DB/script_level/dataset/vocab.json')
    utilities.write_json(vocab_path, vocab)

def main():
    
    
    datasets_path = Path(Path.cwd(), 'create_DataBase/DB/script_level/dataset')
    print(datasets_path)
    datasets = utilities.get_files_in_a_directory(datasets_path)

    
    # load dataset and process scripts
    # for dataset in datasets:
    #     vocab_path = Path(Path.cwd(), 'create_DataBase/DB/script_level/dataset/vocab.json')
    #     with open(vocab_path, 'r') as file:
    #         vocab = json.load(file)
    #     dataset_name = dataset.split('/')[-1]    
    #     if dataset_name.startswith("DB_"):
    #         print(f"Loading {dataset_name}")
    #         with open(dataset) as f:
    #             df = pd.read_csv(f)
    #             process_scripts(df, vocab, dataset_name)

    # concatenate datasets
    revised_dataset_path = Path(Path.cwd(), 'create_DataBase/DB/script_level/dataset/sticked_functions/filtered')
    revised_datasets = utilities.get_files_in_a_directory(revised_dataset_path)

    all_dataframes = []
    for filename in revised_datasets:
        if filename.endswith(".csv"):
                print(filename)
                # Read the CSV file contained within the zip file
                with open(filename) as f:
                    df = pd.read_csv(f)
                    print(df.shape[0])
                    all_dataframes.append(df)
    print("concatenating")
    full_dataframe = pd.concat(all_dataframes, ignore_index=True)
    print(full_dataframe.shape[0])
    print("saving in process ...")
    full_dataframe.to_csv(Path(revised_dataset_path, 'concated_script_level_dataset.csv'), index=False)
    print("done!")
                    
if __name__ == "__main__":
    main()