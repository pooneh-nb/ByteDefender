

# def DPCNN(model,bytecode_sequences, labels, model_name):
#     doc_vectors = np.array([document_vector(model, doc) for doc in bytecode_sequences])

#     X_train, X_test, y_train, y_test = train_test_split(doc_vectors, labels, test_size=0.2, random_state=42)

#     if model_name == 'w2v':
#         # embedding_size is the same as the vector_size parameter used during training
#         embedding_size = model.vector_size
#         # vocabulary_size can be obtained from the trained model's vocabulary
#         vocabulary_size = len(model.wv.index_to_key)
#     if model_name == 'ft':
#         # This gets the word vectors
#         weights = model.wv.get_normed_vector()
#         vocabulary_size, embedding_size = weights.shape

#     dpcnn_model = tf.keras.models.Sequential()
#     # Embedding layer
#     dpcnn_model.add(Embedding())

#     dpcnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
#     dpcnn_model.fit(X_train, y_train, epochs=10, validation_split=0.1)

#     loss, accuracy = dpcnn_model.evaluate(X_test, y_test)
#     print(f'Accuracy with DCPNN: {accuracy}')