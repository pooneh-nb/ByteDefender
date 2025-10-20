from gensim.models import Word2Vec, FastText
import pandas as pd
from pathlib import Path
import time
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
import logging
import time
import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def Word2Vec_embeddings(bytecode_sequences, meta_hyper):
    print('Word2Vec_embeddings')
    
    # Initialize Word2Vec model
    embed_model = Word2Vec(vector_size=meta_hyper['vector_size'],
                           window=meta_hyper['window'],
                           min_count=meta_hyper['min_count'],
                           workers=meta_hyper['CPU'])
    
    # Build vocabulary from the bytecode sequences
    embed_model.build_vocab(bytecode_sequences)

    # Train the model
    start = time.time()
    print(start)
    embed_model.train(bytecode_sequences, total_examples=len(bytecode_sequences), epochs=meta_hyper['epochs'])  
    end = time.time()
    print(end)
    print(f"Training took {end - start} seconds")

    # Save the trained model
    model_filename = f'word2vec_{meta_hyper["vector_size"]}_{meta_hyper["window"]}_{meta_hyper["epochs"]}.mdl'
    save_path = str(Path.home().joinpath(Path.home().cwd(), 'ML_Model/models', model_filename))
    embed_model.save(save_path)

def FastText_embeddings(bytecode_sequences, meta_hyper):
    print('FastText_embeddings')
    # model_ft = FastText(bytecode_sequences, vector_size=100, window=3, min_count=1, sg=1)
    # return model_ft
    embed_model = FastText(vector_size=meta_hyper['vector_size'], 
                           window=meta_hyper['window'], 
                           min_count = meta_hyper['min_count'], 
                           workers=meta_hyper['CPU'])
    embed_model.build_vocab(bytecode_sequences)

    start = time.time()
    print(start)
    embed_model.train(bytecode_sequences, total_examples=len(bytecode_sequences), epochs=meta_hyper['epochs'])  
    end = time.time()
    print(f"Training took {end - start} seconds")

    model_filename = f'fasttext_{meta_hyper["vector_size"]}_{meta_hyper["window"]}_{meta_hyper["epochs"]}.mdl'
    save_path = str(Path.home().joinpath(Path.home().cwd(), 'ML_Model/models', model_filename))
    embed_model.save(save_path)

def main(vector_s, embedding_model, window, epochs, cpu):
    bytecode_sequences = utilities.read_json(Path.home().joinpath(Path.home().cwd(), 
                                            'create_DataBase/DB/dataset/bytecode_sequences.json'))
    meta_hyper = {
    "vector_size": vector_s,  # size of embedding 
    "window": window,         # context window size
    "min_count": 1,      # minimum word count threshold
    "epochs": epochs,       # number of training epochs
    "CPU": cpu             # number of CPU cores to use
}
    if embedding_model == 'word2vec':
        Word2Vec_embeddings(bytecode_sequences, meta_hyper)
    if embedding_model == 'fasttext':
        FastText_embeddings(bytecode_sequences, meta_hyper)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define Arguments
    parser.add_argument('--vec', type=int, help='Vector size. Options: 50 / 100 / 200', default=50)
    parser.add_argument('--mdl', type=str, help='Embedding method. Options: word2vec, fasttext', default='word2vec')
    parser.add_argument('--win', type=int, help='window. Options: 3 / 5', default=3)
    parser.add_argument('--epc', type=int, help='epochs. Options: 100 / 150', default=5)
    parser.add_argument('--cpu', type=int, help='CPU', default=4)
    
    args = parser.parse_args()
    main(args.vec, args.mdl, args.win, args.epc, args.cpu)
