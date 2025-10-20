from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve, auc, roc_curve
from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
from joblib import load
import matplotlib.pyplot as plt



def load_RF_model():
    best_rf_w2v_model_path = str(Path.home().joinpath(Path.home().cwd(), 'ML_Model/models/RF_word2vec_100_5_300.joblib'))
    best_rf_w2v_model = load(best_rf_w2v_model_path)
    return best_rf_w2v_model

def vectorize_text(model, bytecode):
    # Convert text to tokens and get embeddings, averaging them
    vectors = [model.wv[token] for token in bytecode if token in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def load_datasets():
    w2v_embed_model_path = str(Path.home().joinpath(Path.home().cwd(), f'ML_Model/models/word2vec_100_5_300.mdl'))
    w2v_embed_model = Word2Vec.load(w2v_embed_model_path)

    # prepare data
    bytecode_sequences = utilities.read_json(Path.home().joinpath(Path.home().cwd(), 
                                            'create_DataBase/DB/dataset/bytecode_sequences.json'))

    X = np.array([vectorize_text(w2v_embed_model, bytecode) for bytecode in bytecode_sequences])
    y = np.loadtxt(Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/dataset/labels.json'), delimiter=',')
    return X, y

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
    fpr, tpr, thresholds = roc_curve(y, y_pred)

    print(f"{'-'*30}\nEvaluating on the {dataset_name} set\n{'-'*30}")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"F1 Score    : {f1:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"AUC         : {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    print(f"{'-'*30}\n")

    plot_roc_curve(fpr, tpr, thresholds)

def plot_roc_curve(fpr, tpr, thresholds):
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line for reference
    plt.legend()
    plt.show()
    print(thresholds)

def main():

    # Load X and y
    X, y = load_datasets()

    # Split Temp into Train, Validation and Test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    # Over Sampling
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Upload the best model
    model = load_RF_model()

    # Evaluate on the validation set
    evaluate_model(model, X_val, y_val, "Validation")
    # Evaluate on the test set
    evaluate_model(model, X_test, y_test, "Test")


if __name__ == "__main__":
    main()