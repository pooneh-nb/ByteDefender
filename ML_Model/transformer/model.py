import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall, AUC


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Ensure your custom modules are compatible with tensorflow.keras
from TokenAndPositionEmbedding import TokenAndPositionEmbedding
from TransformerBlock import TransformerBlock

import numpy as np
from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities

import glob

# def batch_generator(sequences, frames, parameters, registers, labels):
#     while True:
#         for seq_batch, label_batch,frame_batch, param_batch, reg_batch  in zip(sequences,frames, parameters, registers, labels):
#             # Combine additional features into a single batch
#             features_batch = np.hstack([frame_batch.reshape(-1, 1), param_batch.reshape(-1, 1), reg_batch.reshape(-1, 1)])
#             # yield {'sequence_input': seq_batch, 'features_input': features_batch}, label_batch
#             yield {'sequence_input': seq_batch, 'features_input': features_batch}, label_batch


# def transformer_model(train_generator, val_generator, steps_per_epoch, validation_steps, vocab_size, maxlen, labels_test):
#     embed_dim = 128
#     num_heads = 5
#     ff_dim = 256
#     feature_dim = 3

#     model_output_path = Path.home().joinpath(Path.cwd(),'ML_Model/transformer/data/model')

#     # The input layer
#     # Sequence input layer
#     sequence_input = layers.Input(shape=(maxlen,), dtype='int32', name='sequence_input')
    
#     # Features input layer, adjust the shape based on your features
#     features_input = layers.Input(shape=(feature_dim,), name='features_input')


#     # Create  the embedding layer instance for sequence input
#     embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
#     x = embedding_layer(sequence_input)

#     # Add additional layers as needed 
#     transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

#     x = transformer_block(x)
#     x = layers.GlobalAveragePooling1D()(x)

#     # Optionally concatenate features input with transformer output here
#     combined = layers.Concatenate()([x, features_input])

#     x = layers.Dropout(0.1)(combined)
#     x = layers.Dense(20, activation="relu")(x)
#     x = layers.Dropout(0.1)(x)
#     outputs = layers.Dense(1, activation="sigmoid")(x)

#     # Build and compile the model
#     model = Model(inputs=[sequence_input, features_input], outputs=outputs)
#     model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", Precision(), Recall(), AUC()])

#     # train the model on your data
#     model.fit(train_generator, steps_per_epoch=steps_per_epoch, validation_data=val_generator, validation_steps=validation_steps, epochs=30)
#     model.save(model_output_path, save_format='tf')

#     # to load the model:
#     # loaded_model = tf.keras.models.load_model('path/to/location') # from tensorflow.keras.models import load_model


# def load_batches(tokenized_padded_path, frame_arrays, parameter_arrays, register_arrays, label_arrays, batch_size):
#     print("load_batches")
#     # Load all tokenized and padded batch files sorted by name
#     batches_files = sorted(utilities.get_files_in_a_directory(tokenized_padded_path))
#     sequence_batches = [np.load(file) for file in batches_files]

#     # split frames into batches
#     frames_batches = [frame_arrays[i:i + batch_size] for i in range(0, len(frame_arrays), batch_size)]
#     # split parameters into batches
#     parameters_batches = [parameter_arrays[i:i + batch_size] for i in range(0, len(parameter_arrays), batch_size)]
#     # split registers into batches
#     registers_batches = [register_arrays[i:i + batch_size] for i in range(0, len(register_arrays), batch_size)]
#     # Split labels into batches
#     labels_batches = [label_arrays[i:i + batch_size] for i in range(0, len(label_arrays), batch_size)]
    

#     # Ensure the number of labels batches matches the number of sequence batches
#     print(len(sequence_batches), len(frames_batches), len(parameters_batches), len(registers_batches), len(labels_batches))
#     assert len(sequence_batches) == len(frames_batches) == len(parameters_batches) == len(registers_batches) == len(labels_batches), "All features and labels must have the same number of batches."
#     return sequence_batches, frames_batches, parameters_batches, registers_batches, labels_batches

# def main():
#     vocab_path = Path.home().joinpath(Path.cwd(),'ML_Model/transformer/data/vocab.txt')                                       
#     vocab = utilities.read_file_splitlines(vocab_path)
#     vocab_size = len(vocab)

#     batch_size = 20

#     bytecode_sequences = utilities.read_json(Path.home().joinpath(Path.cwd(), 
#                                             'create_DataBase/DB/sliced_dataset/bytecode_sequences.json'))

#     sequence_lengths = [len(seq) for seq in bytecode_sequences]
#     # print("#sequences:", len(sequence_lengths))
#     max_length = max(sequence_lengths)

#     # print("max_length:" , max_length)

#     # load lables
#     json_labels = utilities.read_json(Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/sliced_dataset/labels.json'))
#     label_arrays = np.array(json_labels)
    
#     # load frame_size
#     json_frame_sizes = utilities.read_json(Path(Path.cwd(), 'create_DataBase/DB/sliced_dataset/frame_size.json'))
#     frame_arrays = np.array(json_frame_sizes)
    
#     # load parameter_count
#     json_parameter_counts = utilities.read_json(Path(Path.cwd(), 'create_DataBase/DB/sliced_dataset/parameter_count.json'))
#     parameter_arrays = np.array(json_parameter_counts)

#     # load register_count
#     json_register_counts = utilities.read_json(Path(Path.cwd(), 'create_DataBase/DB/sliced_dataset/register_count.json'))
#     register_arrays = np.array(json_register_counts)


#     tokenized_padded_path = Path(Path.cwd(), 'ML_Model/transformer/data/padded_batches')

#     sequences, frames, parameters, registers, labels = load_batches(tokenized_padded_path, frame_arrays, parameter_arrays, register_arrays, label_arrays, batch_size)
    
#     # separate train and test data
#     sequences_train, sequences_test, frames_train, frames_test, parameters_train, parameters_test, registers_train, registers_test, labels_train, labels_test = train_test_split(sequences, frames, parameters, registers, labels, test_size=0.2, random_state=42)
#     sequences_train, sequences_test, labels_train, labels_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

#     train_generator = batch_generator(sequences_train, frames_train, parameters_train, registers_train, labels_train)
#     test_generator = batch_generator(sequences_test, frames_test, parameters_test, registers_test, labels_test)

#     steps_per_epoch = len(sequences_train)     
#     validation_steps = len(sequences_test) 

#     transformer_model(train_generator, test_generator, steps_per_epoch, validation_steps, vocab_size, max_length, labels_test)

# if __name__ == "__main__":
#     main()