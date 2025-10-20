import sys
import random
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall, AUC


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

model_dir = Path(Path.cwd(), 'ML_Model/longformer/model')
sys.path.append(str(model_dir))
sys.path.insert(1, Path.cwd().as_posix())


from ML_Model.longformer.model.TokenAndPositionEmbedding import TokenAndPositionEmbedding
from ML_Model.longformer.model.TransformerWithLocalAttention import TransformerWithLocalAttention

import utilities

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, sequences, labels, batch_size=20, shuffle=True):
        self.sequences = sequences
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
    
        return int(np.ceil(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of sequences for the current batch
        sequences_batch = [self.sequences[k] for k in indexes]
        labels_batch = [self.labels[k] for k in indexes]

        # Generate data
        # return np.array(sequences_batch), np.array(labels_batch).reshape(-1, 1)
        X, y = self.__data_generation(sequences_batch, labels_batch)

        return np.array(X)[0], np.array(y).reshape(-1,1)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.sequences))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, sequences_batch, labels_batch):
        # Generate data
        X = sequences_batch
        y = labels_batch

        return X, y

def my_batch_generator(sequences, labels):
    # print("amghezi")
    while True:
        for seq_batch, label_batch in zip(sequences, labels):
            # Combine additional features into a single batch
            yield {'sequence_input': seq_batch}, label_batch

def build_and_compile_model(max_length, vocab_size):
    embed_dim = 128
    ff_dim = 256
    block_size = 1460
    left_context = 250 
    right_context = 250

    
    # The input layer
    sequence_input = layers.Input(shape=(max_length,), dtype='int32', name='sequence_input')

    # Create  the embedding layer instance for sequence input
    embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim)
    x = embedding_layer(sequence_input)
    # print("embeddings", x.shape)
    
    # Add additional layers as needed 
    # print("embed_dim, block_size, left_context, right_context, ff_dim", embed_dim, block_size, left_context, right_context, ff_dim)
    longformer_block = TransformerWithLocalAttention(embed_dim, block_size, left_context, right_context, ff_dim)

    x = longformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Optionally concatenate features input with transformer output here
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    # Build and compile the model
    model = Model(inputs=sequence_input, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", Precision(), Recall(), AUC()])
    return model
    

def longformer_model(train_generator, val_generator, steps_per_epoch, validation_steps, vocab_size, max_length, batch_size, js_type):

    model_output_path = Path(Path.cwd(),f'ML_Model/longformer/{js_type}/data/model')
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        # Place the model building and compiling inside the strategy scope
        # This ensures the model is mirrored across the GPUs
        model = build_and_compile_model(max_length, vocab_size)

    # train the model on your data
    model.summary()
    print("Start fitting")
    model.fit(train_generator, steps_per_epoch=steps_per_epoch, 
              validation_data=val_generator, validation_steps=validation_steps, epochs=30)
    model.save(model_output_path, save_format='tf')

    # to load the model:
    # loaded_model = tf.keras.models.load_model('path/to/location') # from tensorflow.keras.models import load_model


def load_batches(tokenized_padded_path, label_arrays, batch_size):
    print("load_batches")

    batches_files = sorted(utilities.get_files_in_a_directory(tokenized_padded_path))
    # Generate a list of all batch indices
    batch_indices = list(range(len(batches_files)))
    
    sequence_batches = [np.load(file) for file in batches_files]
    # Randomly select 5 unique batch indices
    selected_batch_indices = random.sample(batch_indices, 3)
    selected_batch_indices.sort()  # Sorting is not necessary but helps in understanding output

    selected_sequence_batches = [sequence_batches[i] for i in selected_batch_indices]

    

    print("Selected Batch Indices:", selected_batch_indices)

    # Generate label batches corresponding to the selected batch indices
    selected_labels_batches = [label_arrays[i * batch_size:(i + 1) * batch_size] for i in selected_batch_indices]

    print("Selected Labels Batches Shapes:")
    for batch in selected_labels_batches:
        print(batch.shape)  # Each should print (20,)
    
    assert len(selected_sequence_batches) == len(selected_labels_batches), "All features and labels must have the same number of batches."

    return selected_sequence_batches, selected_labels_batches


def main():
    js_type = ["raw_scripts", "ggl_obfuscated_scripts", "js_obfuscated_scripts"]
    i = 0
    vocab_path = Path.home().joinpath(Path.cwd(),f'ML_Model/longformer/data/{js_type[i]}/vocab.txt')                                       
    vocab = utilities.read_file_splitlines(vocab_path)
    vocab_size = len(vocab)    
    
    # load lables
    json_labels = utilities.read_json(Path(Path.cwd(), 
                                            f'create_DataBase/DB/sliced_dataset/script_level_DB/{js_type[i]}/labels.json'))
    label_arrays = np.array(json_labels)
    
    tokenized_padded_path = Path(Path.cwd(), f'ML_Model/longformer/data/{js_type[i]}/padded_batches')

    batch_size = 20
    sequences, labels = load_batches(tokenized_padded_path, label_arrays, batch_size)

    max_length = len(sequences[0][0])
    print("max_length:" , max_length)
    
    # separate train and test data
    sequences_train, sequences_test, labels_train, labels_test = \
        train_test_split(sequences, labels, test_size=0.2, random_state=42)
    
    train_generator = my_batch_generator(sequences_train, labels_train)
    test_generator = my_batch_generator(sequences_test, labels_test)
    
    
    steps_per_epoch = len(sequences_train)     # number of batches
    validation_steps = len(sequences_test) 

    longformer_model(train_generator, test_generator, steps_per_epoch, validation_steps, 
                     vocab_size, max_length, batch_size, js_type[i])

if __name__ == "__main__":
    main()