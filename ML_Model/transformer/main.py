# from pathlib import Path
# import sys
# sys.path.insert(1, Path.home().cwd().as_posix())
# import utilities

# import data_tokenizer
# import ML_Model.transformer.model_old as model_old
# import numpy as np

# def main():
    

#     ## prepare dataset
#     padded_sequences = np.load(tokenized_dataset_path)
#     json_labels = utilities.read_json(Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/dataset/labels.json'))
#     labels = np.array(json_labels)
    
#     maxlen = len(padded_sequences[0])
    
#     model_old.transformer_model(vocab_path, padded_sequences, labels, maxlen)


# if __name__ == "__main__":
#     main()