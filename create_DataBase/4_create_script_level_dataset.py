import sys
import json
from pathlib import Path
sys.path.insert(1, Path.cwd().as_posix())
import utilities
import pandas as pd
import ast

print("Loading dataset")
dir = Path(Path.cwd(), 'create_DataBase/DB/script_level/dataset/script_level_dataset.csv')
df_scripts = pd.read_csv(dir)

print("Removing NAN")
df_scripts.dropna(inplace=True)

print("Removing duplicates")
df_scripts.drop_duplicates(inplace=True)

# Invert the vocabulary to map numbers to tokens
dict_path = Path(Path.cwd(), 'create_DataBase/DB/script_level/dataset/vocab.json')
with open(dict_path, 'r') as file:
    vocab = json.load(file)

inv_vocab = {int(v): k for k, v in vocab.items()}

print("converting string bytecodes into list and then numbers to tokens")
# Safe conversion from string representation of list to actual list of integers
df_scripts['Bytecode'] = df_scripts['Bytecode'].apply(ast.literal_eval)

# Vectorized mapping operation using a list comprehension directly
df_scripts['Bytecode'] = df_scripts['Bytecode'].apply(lambda x: [inv_vocab[i] for i in x])

output_path = Path(Path.cwd(), 'create_DataBase/DB/sliced_dataset/script_level.csv')

print("Buffering...")
df_scripts.to_csv(output_path, index=False)