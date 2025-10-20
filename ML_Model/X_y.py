from pathlib import Path
import numpy as np
import pandas as pd
import ast
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities


# db1_path = Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB_batch1/dataset/bytecode_data.csv')
# db1 = pd.read_csv(db1_path, header=None)
# print(db1.shape[0])
# db2_path = Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB_batch2/dataset/FP_bytecode_data.csv')
# db2 = pd.read_csv(db2_path, header=None)
# print(db2.shape[0])

# db = pd.concat([db1, db2], ignore_index=True)

# columns = ['Site', 'ScriptID', 'FunctionName', 'ParameterCount', 'RegisterCount', 'FrameSize', 'Bytecode', 'Label']

db_path = Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/dataset/bytecode_data.csv')
db = pd.read_csv(db_path, header=None)
print(db.shape[0])

# harmonic dtypes
db[6] = db[6].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
# Convert numeric, coercing errors to NaN
db[3] = pd.to_numeric(db[3], errors='coerce')
db[4] = pd.to_numeric(db[4], errors='coerce')
db[5] = pd.to_numeric(db[5], errors='coerce')

# Optional: Fill NaN values with a default value, e.g., 0, if it makes sense for your data
db[3] = db[3].fillna(0).astype(int)
db[4] = db[4].fillna(0).astype(int)
db[5] = db[5].fillna(0).astype(int)

bytecode_sequences = db[6].to_list()
print(len(bytecode_sequences))
parameter_count = db[3].values.tolist()
print(len(parameter_count))
register_count = db[4].values.tolist()
print(len(register_count))
frame_size = db[5].values.tolist()
print(len(frame_size))
label = db[7].values.tolist()
print(len(label))


print(db.shape[0])

utilities.write_json(Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/dataset/bytecode_sequences.json'), bytecode_sequences)
utilities.write_json(Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/dataset/labels.json'), label)
utilities.write_json(Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/dataset/parameter_count.json'), parameter_count)
utilities.write_json(Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/dataset/register_count.json'), register_count)
utilities.write_json(Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/dataset/frame_size.json'), frame_size)

