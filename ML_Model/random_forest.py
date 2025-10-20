"""
The ideal way to incorporate SMOTE with hyperparameter tuning is to ensure that SMOTE is applied within each cross-validation fold. 
This means that for each fold in the cross-validation process, SMOTE is applied only to the training part of that fold.
This can be achieved by creating a pipeline that first applies SMOTE and then trains the model. 
This pipeline is then used within the cross-validation process.
"""
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve, auc
from gensim.models import Word2Vec, FastText, Doc2Vec
from joblib import dump

import numpy as np
from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
import time
import argparse



def vectorize_text(model, bytecode, model_type):
    """
    Converts a list of tokens (bytecode) into an embedding vector.

    :param model: The embedding model (Doc2Vec or Word2Vec/FastText).
    :param bytecode: A list of tokens to be vectorized.3
    :param model_type: A string indicating the type of model ('doc2vec' or 'word2vec').

    :return: A single vector representing the bytecode.
    """

    # For Doc2Vec model
    if model_type == 'doc2vec':
        # 'bytecode' is a list of tokens for Doc2Vec
        return model.infer_vector(bytecode)
    
    # For Word2Vec or FastText model
    elif model_type in ['word2vec', 'fasttext']:
        vectors = [model.wv[token] for token in bytecode if token in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

    else:
        raise ValueError("Invalid model type specified. Choose 'doc2vec', 'word2vec', or 'fasttext'.")

def evaluate_model(model, X, y, dataset_name):
    """
    Function to evaluate the model

    """
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='binary')
    precision = precision_score(y, y_pred, average='binary')
    recall = recall_score(y, y_pred, average='binary')
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred)
    auc_score = auc(recall_curve, precision_curve)

    print(f"{'-'*30}\nEvaluating on the {dataset_name} set\n{'-'*30}")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"F1 Score    : {f1:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"AUC         : {auc_score:.4f}")
    print("\nClassification Report:")
    # print(classification_report(y, y_pred))
    print(f"{'-'*30}\n")

def random_forest(X, y, embed_model, vector_size, window, epochs, cpu):

    print(f"Random Forest with {embed_model} embeddings")
    # Initial split: Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    print("apply smote")
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    rf_params = {
        'n_estimators': [100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [10, 20, 30],
        'min_samples_split': [5, 10],
        'criterion': ['gini', 'entropy']
    }
    
    # Set up the RandomForestClassifier
    print("Set up the RandomForestClassifier")
    rf = RandomForestClassifier(random_state=42)

    print('Initialize GridSearchCV') 
    # Initialize GridSearchCV
    grid_search = GridSearchCV(rf, param_grid=rf_params, cv=5, n_jobs=cpu, verbose=2)

    # Train the model using GridSearchCV on the training data
    print("Train the model using GridSearchCV on the training data")
    start = time.time()
    print(start)
    grid_search.fit(X_train_smote, y_train_smote)
    end  = time.time()
    print(end)

    print(f"Training took {end - start} seconds")
    # Best hyperparameters
    
    print("Best Parameters:", grid_search.best_params_)

    # extract the best model
    best_rf = grid_search.best_estimator_
    model_filename = f'RF_{embed_model}_{vector_size}_{window}_{epochs}.joblib'
    save_path = str(Path.home().joinpath(Path.home().cwd(), 'ML_Model/models', model_filename))
    dump(best_rf, save_path)

    # Evaluate on the test set
    evaluate_model(best_rf, X_test, y_test, "Test")


def main(embed_model, vector_size, window, epochs, cpu):
    # load fasttext embedding model
    embed_model_path = str(Path.home().joinpath(Path.home().cwd(), f'ML_Model/models/{embed_model}_{vector_size}_{window}_{epochs}.mdl'))
    em_model = None
    if embed_model == 'word2vec':
        em_model = Word2Vec.load(embed_model_path)
    if embed_model == 'fasttext':
        em_model = FastText.load(embed_model_path)
    if embed_model == 'doc2vec':
        em_model = Doc2Vec.load(embed_model_path)    

    # prepare data
    bytecode_sequences = utilities.read_json(Path.home().joinpath(Path.home().cwd(), 
                                            'create_DataBase/DB/dataset/bytecode_sequences.json'))
    parameter_count = np.array(utilities.read_json(Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/dataset/parameter_count.json')))[:, np.newaxis]
    register_count = np.array(utilities.read_json(Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/dataset/register_count.json')))[:, np.newaxis]
    frame_size = np.array(utilities.read_json(Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/dataset/frame_size.json')))[:, np.newaxis]

    print("Prepare X and y")
    bytecode_embeddings = np.array([vectorize_text(em_model, bytecode, embed_model) for bytecode in bytecode_sequences])
    print(bytecode_embeddings.shape)
    print(parameter_count.shape)
    print(register_count.shape)
    print(frame_size.shape)
    
    X = np.hstack((bytecode_embeddings, parameter_count, register_count, frame_size))
    print(X.shape)
    y = utilities.read_json(Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/dataset/labels.json'))
    random_forest(X, y, embed_model, vector_size, window, epochs, cpu)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Defining Arguments
    parser.add_argument('--mdl', type=str, 
                        help='embedding model. options: word2vec / fasttext / doc2vec', default='doc2vec')
    parser.add_argument('--vec', type=int, help='vector_size. options: 50 / 100', default=50)
    parser.add_argument('--win', type=int, help='window. options: 3 / 5', default=3)
    parser.add_argument('--epc', type=int, help='epochs. options: 150 / 200 / 300', default=5)
    parser.add_argument('--cpu', type=int, help='CPU', default=4)

    # parsing arguments
    args = parser.parse_args()

    main(args.mdl, args.vec, args.win, args.epc, args.cpu)