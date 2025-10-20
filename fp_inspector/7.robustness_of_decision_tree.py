import pandas as pd
import ast
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_score, recall_score
import argparse

def load_and_prepare_data(filepath):
    """Load data from CSV, parse lists, and create features DataFrame."""
    df = pd.read_csv(filepath)
    df['features_vector'] = df['features_vector'].apply(ast.literal_eval)
    features_df = pd.DataFrame(df['features_vector'].tolist())
    full_df = pd.concat([df[['script_name', 'label']], features_df], axis=1)
    unique_df = full_df.drop_duplicates(subset=features_df.columns.tolist())
    print(f"Data loaded and prepared from {filepath}, unique_df shape: {unique_df.shape}")
    return unique_df

def balance_dataset(df):
    """Balance dataset based on the 'label' column."""
    df_label_1 = df[df['label'] == 1]
    df_label_0 = df[df['label'] == 0]
    min_count = min(len(df_label_1), len(df_label_0))
    balanced = pd.concat([
        df_label_1.sample(n=min_count, random_state=42),
        df_label_0.sample(n=min_count, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Dataset balanced, shape: {balanced.shape}")
    return balanced

def main(js_type):
    # Paths to datasets
    path_raw_js = Path.cwd() / 'fp_inspector/data/raw_js_reduced_dim_input_features.csv'
    path_obfs_js = Path.cwd() / f'obfuscation/data/{js_type}_reduced_dim_input_features.csv'

    # Load and prepare data
    unique_df = load_and_prepare_data(path_raw_js)
    unique_df_obfs = load_and_prepare_data(path_obfs_js)
    print(unique_df_obfs.shape)

    # Balance obfuscated dataset
    balanced_df_obfs = balance_dataset(unique_df_obfs)
    print(balanced_df_obfs.shape)

    # Prepare labels and features
    y = unique_df['label'].values
    X = unique_df.drop(columns=['script_name', 'label'])
    y_obfs = balanced_df_obfs['label'].values
    X_obfs = balanced_df_obfs.drop(columns=['script_name', 'label'])

    # Train-test split for internal validation (optional here as main test set is separate)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree classifier
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Predictions and evaluations
    y_pred = classifier.predict(X_obfs)
    y_probs = classifier.predict_proba(X_obfs)[:, 1]
    report = classification_report(y_obfs, y_pred, target_names=['Class 0', 'Class 1'])
    accuracy = accuracy_score(y_obfs, y_pred)
    precision = precision_score(y_obfs, y_pred, average='binary')
    recall = recall_score(y_obfs, y_pred, average='binary')
    auc_score = roc_auc_score(y_obfs, y_probs)

    # Output results
    print(report)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"AUC: {auc_score:.2f}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Process JavaScript files to generate ASTs.")
    # parser.add_argument('js_type', type=str, help="The subdirectory within 'obfuscated_js' to use")
    
    # args = parser.parse_args()
    # main(args.js_type)
    main('beautifytools')