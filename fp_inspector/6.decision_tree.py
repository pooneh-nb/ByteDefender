import pandas as pd
import ast
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_score, recall_score, average_precision_score


# Define the path to the CSV file
df_path = Path.cwd() / 'fp_inspector/data/raw_js_reduced_dim_input_features.csv'
df = pd.read_csv(df_path)

# Convert the 'features_vector' from string representation of lists to actual lists
df['features_vector'] = df['features_vector'].apply(ast.literal_eval)


# Create a DataFrame of features where each element of each list becomes a column
features_df = pd.DataFrame(df['features_vector'].tolist())

# Combine the features_df back with script_name and label for a complete DataFrame
full_df = pd.concat([df[['script_name', 'label']], features_df], axis=1)

# Drop duplicates based on all feature columns
unique_df = full_df.drop_duplicates(subset=features_df.columns.tolist())


# Labels
y = unique_df['label'].values

X = unique_df.drop(columns=['script_name', 'label'])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the Decision Tree classifier
classifier = DecisionTreeClassifier(random_state=42)

# Fitting the classifier to the training data
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_probs = classifier.predict_proba(X_test)[:, 1]

# Generating the classification report and calculating accuracy
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Generating the classification report
report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
auc_score = roc_auc_score(y_test, y_probs)

# Threshold-agnostic metrics
roc_auc = roc_auc_score(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)



print(report)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"AUC: {auc_score}")
print(f"ROC AUC: {roc_auc:.2f}")
print(f"PR AUC: {pr_auc:.2f}")