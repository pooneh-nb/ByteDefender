from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
import numpy as np
import json
from collections import OrderedDict
import tensorflow as tf
import tensorflow_text as text
import ast

from concurrent.futures import ThreadPoolExecutor

def edit_bytecodes(bytecode_sequences):
    print("edit_bytecodes")
    bytecode_list = []
    for bytecode in bytecode_sequences:
        print(type(bytecode))
        bytecode = ast.literal_eval(bytecode)
        bytecode_list.append(bytecode)
    
    bytecode_path = Path(Path.cwd(), 'create_DataBase/DB/sliced_unique_dataset/bytecode_sequences.json')
    with open(bytecode_path, 'w') as file:
        json.dump(bytecode_list, file, indent=4)
        


def get_vocab(bytecode_sequences, vocab_path):
    print("get_vocab")

    vocab = set()
    for bytecode in bytecode_sequences:
        for word in bytecode:
            vocab.add(word)


    # Define special tokens
    special_tokens = ['[PAD]', '[CLS]', '[EOS]', '[UNK]']

    # Add special tokens at the beginning of the vocabulary
    vocab_sorted = special_tokens + sorted(list(vocab))

    print(f"Vocabulary size including special tokens: {len(vocab_sorted)}")

    # Write the vocabulary to a text file
    with open (vocab_path, 'w') as vocab_file:
        for token in vocab_sorted:
            vocab_file.write(token + '\n')

def pad_to_max_length(tensor, max_length, pad_value):
    """
    Pads or truncates a tensor to a specific max_length.
    
    Args:
    - tensor: Input tensor (2D).
    - max_length: The maximum length to pad/truncate to.
    - pad_value: Value used for padding.
    
    Returns:
    - Tensor padded or truncated to max_length.
    """
    # Convert the ragged tensor to a dense tensor, padding with pad_value
    tensor = tensor.to_tensor(default_value=pad_value)
    # Ensure the tensor is only padded up to max_length
    tensor_shape = tf.shape(tensor)
    # tf.cond(pred, true_fn, false_fn)
    tensor = tf.cond(
        tensor_shape[1] > max_length,
        lambda: tensor[:, :max_length],  # Truncate to max_length if longer
        lambda: tf.pad(  # Pad to max_length if shorter
            tensor, [[0, 0], [0, max_length - tensor_shape[1]]], constant_values=pad_value)
    )
    
    return tensor

def tokenize_and_pad_sequences(bytecode_sequences, vocab_table, max_seq_length, cls_id, eos_id, pad_id, tokenized_dataset_path):
    print("entering tokenize_and_pad_sequences")
    batch_size = 512
    # Convert sequences to a RaggedTensor (the length of all sequences will be set to max_length)
    # A RaggedTensor is a type of tensor designed to handle sequences of varying lengths, which is common in sequence processing tasks

    rt = tf.ragged.constant(bytecode_sequences)
    
    # Lookup tokens in the vocabulary table to get their IDs
    print("Lookup tokens in the vocabulary table ")
    token_ids = vocab_table.lookup(rt)
    
    # Add [CLS] and [EOS] tokens
    cls_tokens = tf.fill([token_ids.nrows(), 1], cls_id) # create a vector (tf) of cls_ids to the length of bytecode_sequenses
    eos_tokens = tf.fill([token_ids.nrows(), 1], eos_id)
    token_ids = tf.concat([cls_tokens, token_ids, eos_tokens], axis=1)
    
    # Create a TensorFlow dataset from the token IDs
    print("Create a TensorFlow ")
    dataset = tf.data.Dataset.from_tensor_slices(token_ids)
    dataset = dataset.batch(batch_size)

    # Dynamically pad each batch
    print("padded_batches")
    padded_batches = dataset.map(lambda x: pad_to_max_length(x, max_seq_length, pad_id),
                                num_parallel_calls=tf.data.AUTOTUNE)

    # padded_batches = dataset.map(lambda x: x.to_tensor(default_value=pad_id))

    # Save each padded batch to disk
    # with ThreadPoolExecutor(max_workers=50) as executor:
    print("Saving pads")
    for i, padded_batch in enumerate(padded_batches):
        batch_path = Path.home().joinpath(tokenized_dataset_path, f"batch_{i:07}.npy") 
        np.save(batch_path, padded_batch.numpy())

def prepare_tokenizer(bytecode_sequences, vocab_path, tokenized_dataset_path):
    print("prepare_tokenizer")
    # Load the vocabulary into a lookup table
    vocab_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            vocab_path,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
        ),
        num_oov_buckets=1
    )

    cls_id = vocab_table.lookup(tf.constant("[CLS]")).numpy()
    eos_id = vocab_table.lookup(tf.constant("[EOS]")).numpy()
    pad_id = vocab_table.lookup(tf.constant("[PAD]")).numpy()

    sequence_lengths = [len(seq) for seq in bytecode_sequences]
    max_length = max(sequence_lengths)
    print(max_length)
    tokenize_and_pad_sequences(bytecode_sequences, vocab_table, max_length, cls_id, eos_id, pad_id, tokenized_dataset_path)

    # np.save(tokenized_dataset_path, padded_sequences.numpy())

def main():
    ## build vocabulary
    bytecode_sequences = utilities.read_json(Path(Path.cwd(), 
                                            'create_DataBase/DB/sliced_unique_dataset/bytecode_sequences.json'))

    # convert strings of lists to lists
    # edit_bytecodes(bytecode_sequences)
    
    # Define the path for the vocabulary text file
    vocab_path = Path(Path.cwd(),'ML_Model/transformer/data/vocab.txt')
    # Check if the directory exists, if not, create it
    if not vocab_path.parent.exists():
        print("Directory does not exist, creating...")
        vocab_path.parent.mkdir(parents=True, exist_ok=True)  # parents=True allows creating parent directories
        print(f"Directory created at {vocab_path.parent}")

    # Check if the file exists, if not, create it
    if not vocab_path.exists():
        print("File does not exist, creating...")
        vocab_path.touch()  # Creates the file
        print(f"File created at {vocab_path}")
    else:
        print(f"File already exists at {vocab_path}")
    
    get_vocab(bytecode_sequences, vocab_path)

    # ## tokenize dataset
    tokenized_dataset_path = Path(Path.cwd(), 'ML_Model/transformer/data/padded_batches')
    if not tokenized_dataset_path.exists():
        tokenized_dataset_path.mkdir(exist_ok=True)
    prepare_tokenizer(bytecode_sequences, vocab_path, tokenized_dataset_path)

if __name__ == "__main__":
    main()