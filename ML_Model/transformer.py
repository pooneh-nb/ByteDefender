import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
import numpy as np


tf.get_logger().setLevel('ERROR')


def build_classifier_model():
    tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    
    # Define the input layer
    input_word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_word_ids")
    # The encoder inputs are now directly the input_word_ids
    outputs = encoder({'input_word_ids': input_word_ids, 'input_mask': None, 'input_type_ids': None})
    
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)
    
    return tf.keras.Model(input_word_ids, net)


def convert_to_token_ids(sequences, max_length=512):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_ids =  [tokenizer.convert_tokens_to_ids(seq) for seq in sequences]
    # Pad the sequences
    padded_token_ids = pad_sequences(token_ids, maxlen=max_length, padding='post', truncating='post')
    return padded_token_ids


# wrapping raw dataset with tf.data
def get_dataset(bytecode_sequences, labels, batch_size=32, shuffle=True, buffer_size=10000):
    dataset = tf.data.Dataset.from_tensor_slices((bytecode_sequences, labels))
    if shuffle:
        dataset = dataset.shuffle(len(bytecode_sequences))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def main():
    # load bytecode sequences and lables
    bytecode_sequences = utilities.read_json(Path.home().joinpath(Path.home().cwd(), 
                                                'create_DataBase/DB/dataset/bytecode_sequences.json'))
    labels = utilities.read_json(Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/dataset/labels.json'))

    X_train, X_test, y_train, y_test = train_test_split(bytecode_sequences, labels, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    X_train_ids = convert_to_token_ids(X_train)
    X_test_ids = convert_to_token_ids(X_test)

    # Convert sequences and labels to a TensorFlow dataset
    train_data = get_dataset(X_train_ids, y_train)
    test_data = get_dataset(X_test_ids, y_test)

    classifier_model = build_classifier_model()
    classifier_model.compile(optimizer='adam',
                             loss='binary_crossentropy',  # Adjusted to 'binary_crossentropy'
                             metrics=['accuracy'])

    history = classifier_model.fit(train_data, epochs=5, validation_data=test_data)
    eval_results = classifier_model.evaluate(test_data)

if __name__ == "__main__":
    main()