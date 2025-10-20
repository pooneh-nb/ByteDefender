import tensorflow as tf
import numpy as np
from pathlib import Path

def load_batches_into_dataset(tokenized_dataset_path, labels_path, batch_size):
    """
    Loads tokenized and padded batches from disk into a TensorFlow Dataset.
    """
    dataset = tf.data.Dataset.list_files(str(tokenized_dataset_path/'batch_*.npy'))

    def load_batch(batch_path):
        batch = np.load(batch_path.numpy())
        labels = np.load(labels_path.numpy())
        return batch, labels
    
    dataset = dataset.map(lambda x: tf.py_function(load_batch, [x], Tout=tf.int64),
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    
    return dataset