from pathlib import Path
import sys
sys.path.insert(1, Path.cwd().as_posix())
import utilities 
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import argparse
import pandas as pd

from tqdm import tqdm

def reduce_feature_dimensionality(df):
    print("reduce_feature_dimensionality")
    X = df.drop(['label', 'script_name'], axis=1)
    y = df['label']

    # Initialize and fit the RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    forest.fit(X, y)

    # Extract feature importances
    importances = forest.feature_importances_

    # Create a DataFrame of features and their importances
    features = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    })

    # Sort features by their importance in descending order
    top_features = features.sort_values(by='importance', ascending=False).head(1000)

    # Filter the original DataFrame to keep only the top 1,000 features
    reduced_df = df[['script_name', 'label']]

    # Getting the list of top feature names
    feature_columns = top_features['feature'].tolist()

    # Creating a new column that contains the vector of features
    reduced_df.loc[:, 'features_vector'] = df[feature_columns].values.tolist()
    # reduced_df = df[['script_name', 'label'] + top_features['feature'].tolist()]

    return reduced_df
    

def pruning_low_variance_features(df): # unsupervised learning
    print("pruning_low_variance_features")
    X = df.drop(['label', 'script_name'], axis=1)
    
    # Initialize the VarianceThreshold selector
    selector = VarianceThreshold(threshold=0.01)
    
    try:
        # Transform the data
        X_reduced = selector.fit_transform(X)
        
        # If no feature meets the threshold, handle the case
        if X_reduced.size == 0:
            raise ValueError("No feature in X meets the variance threshold 0.01")

        # Get the columns that were kept
        X_reduced_df = pd.DataFrame(X_reduced, columns=X.columns[selector.get_support()])

        # Concatenate the non-feature columns back
        return pd.concat([df[['script_name', 'label']], X_reduced_df], axis=1)

    except Exception as e:
        # Handle the exception if there's an error in the selection process
        print(f"An error occurred: {e}")
        return df  # Return original dataframe or handle differently as needed

def extract_features(script_content, keywords_list):
    return {f'feature_{keyword}': 1 if keyword in script_content else 0 for keyword in keywords_list}

def create_dataset(features_directory, keywords_list):
    print("create_dataset")
    data = []
    feature_files = utilities.get_files_in_a_directory(features_directory)

    count_pos = 0
    count_neg = 0

    for ff in tqdm(feature_files):
        ff_content = utilities.read_file_newline_stripped(ff)
        features = extract_features(ff_content, keywords_list)
        features['script_name'] = ff.split('/')[-1]

        label = ff.split('_')[-1].split('.')[0]
        features['label'] = label

        is_positive = label == '1'
        
        if (is_positive and (count_pos < 0.1 * (count_pos + count_neg + 1))) or (not is_positive and (count_neg < 40 * count_pos + 1)):
            data.append(features)
            # unique_identifiers.add(identifier)
            if is_positive:
                count_pos += 1
            else:
                count_neg += 1

        # data.append(features)
    print(count_pos, count_neg)
    return pd.DataFrame(data)


def main(js_type):
    base_dir = Path(Path.cwd(), 'obfuscation/data')
    features_directory = Path(base_dir, f'api_features/{js_type}')
    js_keywords_file = Path(Path.cwd(), 'fp_inspector/data/cleaned_apis_unique.txt')
    keywords_list = [key_js.strip() for key_js in utilities.read_file_newline_stripped(js_keywords_file)]
    # df = create_dataset(features_directory, keywords_list)
    # df.to_csv(Path(base_dir, f'{js_type}_initial_df.csv'), index=False)
    df = pd.read_csv(Path(base_dir, f'{js_type}_initial_df.csv'))
    df_low_variance_removed = pruning_low_variance_features(df)
    reduced_df = reduce_feature_dimensionality(df_low_variance_removed)
    reduced_df.to_csv(Path(base_dir, f'{js_type}_reduced_dim_input_features.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process JavaScript files to generate ASTs.")
    parser.add_argument('js_type', type=str, help="The subdirectory within 'obfuscated_js' to use")
    
    args = parser.parse_args()
    main(args.js_type)