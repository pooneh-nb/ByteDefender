from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
import logging
import time
import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def pvdm_embeddings(tagged_data, meta_hyper):
    model = Doc2Vec(vector_size=meta_hyper["vector_size"],
                    window=meta_hyper["window"],
                    min_count=meta_hyper["min_count"],
                    workers=meta_hyper["CPU"],
                    dm=1)
    # Build vocabulary
    print("Build vocabulary")
    model.build_vocab(tagged_data)
    
    #Train the model (you can adjust epochs as needed)
    start = time.time()
    print(start)
    print(model.corpus_count)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=meta_hyper["epochs"])
    end = time.time()
    print(f"Training took {end - start} seconds")

    # save the model
    model_filename = f'doc2vec_{meta_hyper["vector_size"]}_{meta_hyper["window"]}_{meta_hyper["epochs"]}.mdl'
    save_path = str(Path.home().joinpath(Path.home().cwd(), 'ML_Model/models', model_filename))
    model.save(save_path)

def main(vectorsize, window, epochs, cpu):
    # load dataset of bytecodes; list[list[words]] 
    bytecode_sequences = utilities.read_json(Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/dataset/bytecode_sequences.json'))

    # Prepare data for Doc2Vec
    tagged_data = [TaggedDocument(words=bytecode, tags=[str(i)]) for i, bytecode in enumerate(bytecode_sequences)]

    meta_hyper = {
        "vector_size": vectorsize, # size of embedding 
        "window": window,      # context window size
        "min_count": 1,      # minimum word count threshold
        "epochs": epochs,      # number of training epochs
        "CPU": cpu         # number of cpus to run in parallel mode
    }

    pvdm_embeddings(tagged_data, meta_hyper)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define Arguments
    parser.add_argument('--vec', type=int, help='vector_size. Options: 50 / 100 / 200', default=50)
    parser.add_argument('--win', type=int, help='window. Options: 3 / 5', default=3)
    parser.add_argument('--epc', type=int, help='epochs. Options: 150 / 200 / 300', default=5)
    parser.add_argument('--cpu', type=int, help='CPU', default=4)
    
    args = parser.parse_args()
    main(args.vec, args.win, args.epc, args.cpu)
