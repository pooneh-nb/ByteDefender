from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities

import keras
from keras import layers

from sklearn.model_selection import train_test_split

import TokenAndPositionEmbedding
import TransformerBlock

def transformer_model(vocab_path, padded_sequences, labels, maxlen_seq):

    vocab = utilities.read_file_splitlines(vocab_path)
    x_train, x_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

    print(len(x_train), "Training sequences")
    print(len(x_test), "Test sequences")

    # Parameters
    vocab_size = len(vocab) 
    maxlen = maxlen_seq  # The uniform length of padded sequences
    # !! distilation 
    embed_dim = 128  # Embedding size for each token 
    num_heads = 6  # Number of attention heads
    ff_dim = 256  # Hidden layer size in feed forward network inside transformer [2X embedding dim]

    # Define the input layer
    inputs = layers.Input(shape=(maxlen,), dtype='int32')

    # Create the embedding layer instance
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)

    # Add additional layers as needed 
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

    # x = transformer_block(x)
    # Stack 4 Transformer Blocks
    for _ in range(4):
        transformer_block = TransformerBlock.TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)

    # !! use the output of last layer
    # it might destroy information    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    # Build and compile the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model on our data
    model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_test, y_test))