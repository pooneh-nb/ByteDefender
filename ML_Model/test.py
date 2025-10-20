from pathlib import Path
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities
from joblib import dump, load
import time


    
w2v_embed_model_path = str(Path.home().joinpath(Path.home().cwd(), 'ML_Model/models/word2vec_100_5_150.mdl'))
w2v_embed_model = Word2Vec.load(w2v_embed_model_path)
bytecode_sequences = utilities.read_json(Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/dataset/bytecode_sequences.json'))
X = np.array([vectorize_text(w2v_embed_model, bytecode) for bytecode in bytecode_sequences])
y = np.loadtxt(Path.home().joinpath(Path.home().cwd(), 'create_DataBase/DB/dataset/labels.json'), delimiter=',')

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Split Temp into Validation and Test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# apply oversampling rare labels with smote
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model_filename = f'RF_w2v_{100}_{5}_{150}.joblib'
saved_rf_model_path = str(Path.home().joinpath(Path.home().cwd(), 'ML_Model/models', model_filename))
best_rf = load(saved_rf_model_path)


# Evaluate on the validation set
print("Evaluate on the validation set")
y_val_pred = best_rf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation F1_score", f1_score(y_val, y_val_pred, average='binary'))
print("Validation Precision", precision_score(y_val, y_val_pred, average='binary'))
print("Validation recall", recall_score(y_val, y_val_pred, average='binary'))
precision, recall, thresholds = precision_recall_curve(y_val, y_val_pred)
print("Validation AUC", auc(recall, precision))
print("\nValidation Classification Report:\n", classification_report(y_val, y_val_pred))


# Evaluate on the test set
print("Evaluate on the test set")
y_test_pred = best_rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Validation F1_score", f1_score(y_test, y_test_pred, average='binary'))
print("Validation Precision", precision_score(y_test, y_test_pred, average='binary'))
print("Validation recall", recall_score(y_test, y_test_pred, average='binary'))
precision, recall, thresholds = precision_recall_curve(y_val, y_test_pred)
print("Validation AUC", auc(recall, precision))
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))