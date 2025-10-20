from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
import numpy as np

import tensorflow as tf
import tensorflow_text as text

# from concurrent.futures import ThreadPoolExecutor

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
    batch_size = 20
    # Convert sequences to a RaggedTensor (the length of all sequences will be set to max_length)
    rt = tf.ragged.constant(bytecode_sequences)
    
    # Lookup tokens in the vocabulary table to get their IDs
    token_ids = vocab_table.lookup(rt)
    
    # Add [CLS] and [EOS] tokens
    cls_tokens = tf.fill([token_ids.nrows(), 1], cls_id)
    eos_tokens = tf.fill([token_ids.nrows(), 1], eos_id)
    token_ids = tf.concat([cls_tokens, token_ids, eos_tokens], axis=1)
    
    # Create a TensorFlow dataset from the token IDs
    dataset = tf.data.Dataset.from_tensor_slices(token_ids)
    dataset = dataset.batch(batch_size)

    # Dynamically pad each batch
    padded_batches = dataset.map(lambda x: pad_to_max_length(x, max_seq_length, pad_id),
                                num_parallel_calls=tf.data.AUTOTUNE)

    # Save each padded batch to disk
    for i, padded_batch in enumerate(padded_batches):
        batch_path = Path.home().joinpath(tokenized_dataset_path, f"batch_{i:03}.npy") 
        # f"{tokenized_dataset_path}_batch_{i}.npy"
        np.save(batch_path, padded_batch.numpy())

def prepare_tokenizer(bytecode_sequences, vocab_path, tokenized_dataset_path):
    print("prepare_tokenizer")
    vocab = utilities.read_file_splitlines(vocab_path)
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

def main():
    for i in range(0,3):
        print("$$$$$$$$$$$$$$$$$$$$$$$$$", i, "$$$$$$$$$$$$$$$$$$$$$$$$$")
        js_type = ["raw_scripts", "ggl_obfuscated_scripts", "js_obfuscated_scripts"]
        ## build vocabulary
        bytecode_sequences = utilities.read_json(
            Path(Path.cwd(), 
                f'create_DataBase/DB/sliced_dataset/script_level_DB/{js_type[i]}/bytecodes.json'))
        # Define the path for the vocabulary text file
        vocab_path = Path.home().joinpath(Path.cwd(),f'ML_Model/longformer/data/{js_type[i]}/vocab.txt')                                       
        get_vocab(bytecode_sequences, vocab_path)

        ## tokenize dataset
        tokenized_dataset_path = Path.home().joinpath(Path.cwd(), f'ML_Model/longformer/data/{js_type[i]}/padded_batches')
        prepare_tokenizer(bytecode_sequences, vocab_path, tokenized_dataset_path)


if __name__ == "__main__":
    main()