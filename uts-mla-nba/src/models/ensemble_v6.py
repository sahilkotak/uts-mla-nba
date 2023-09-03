# Changelog:
# Prev Ver: v3
# Added k-means model to create new feature

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

# Helper function for finding optimal number of clusters
def find_optimal_clusters(data, max_clusters=10):
    from sklearn.metrics import silhouette_score
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k).fit(data)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append((k, score))
    optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
    return optimal_k

def train_stacking_ensemble(train_dataframe, target_column_name, test_size=0.2, random_state=42):
    """
    Train a stacking ensemble of Logistic Regression and SVM, and compute AUROC score on a validation set.

    Parameters:
    - train_dataframe: DataFrame with the preprocessed training data.
    - target_column_name: String, name of the target column.
    - test_size: Fraction of the training data to be used as validation set.
    - random_state: Random seed for reproducibility.

    Returns:
    - Trained Logistic Regression, SVM, and meta-model pipelines.
    - AUROC score on the validation set for the stacking ensemble.
    """

    # Split the preprocessed train data into training and validation sets
    features = train_dataframe.drop(columns=[target_column_name])
    target = train_dataframe[target_column_name]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_size, random_state=random_state)

    # Identify numerical and categorical columns
    numerical_columns = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = features.select_dtypes(include=['object']).columns.tolist()
    
    # Perform k-means clustering to add additional feature
    n_clusters = find_optimal_clusters(features_train[numerical_columns])
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=0).fit(features_train[numerical_columns])
    features_train['kmeans_distance'] = kmeans_model.transform(features_train[numerical_columns]).min(axis=1)
    features_val['kmeans_distance'] = kmeans_model.transform(features_val[numerical_columns]).min(axis=1)
    numerical_columns.append('kmeans_distance')

    # Create a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', StandardScaler(), numerical_columns),
            ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
        ])

    # Create pipelines for both logistic regression and SVM
    logreg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    svm_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='linear', probability=True))
    ])

    # Train both models on the training data
    logreg_pipeline.fit(features_train, target_train)
    svm_pipeline.fit(features_train, target_train)

    # Use trained models to generate predictions on the validation set
    logreg_probabilities = logreg_pipeline.predict_proba(features_val)[:, 1].reshape(-1, 1)
    svm_probabilities = svm_pipeline.predict_proba(features_val)[:, 1].reshape(-1, 1)

    # Stack the predictions to create new features for the meta-model
    stacked_features_val = np.hstack([logreg_probabilities, svm_probabilities])

    # Train the meta-model on the stacked predictions
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(stacked_features_val, target_val)

    # Predict probabilities on the validation set using the meta-model
    ensemble_probabilities = meta_model.predict_proba(stacked_features_val)[:, 1]

    # Compute AUROC score
    auroc_score = roc_auc_score(target_val, ensemble_probabilities)

    return logreg_pipeline, svm_pipeline, meta_model, auroc_score, kmeans_model


def generate_stacking_predictions(logreg_pipeline, svm_pipeline, meta_model, kmeans_model, test_dataframe):
    """
    Generate predictions using the trained stacking ensemble.

    Parameters:
    - logreg_pipeline: Trained Logistic Regression pipeline.
    - svm_pipeline: Trained SVM pipeline.
    - meta_model: Trained meta-model for stacking.
    - kmeans_model: Trained k-means model for adding "kmeans_distance" feature.
    - test_dataframe: DataFrame with the test data.

    Returns:
    - Predictions as a DataFrame.
    """
    
    # Store 'player_id' and then drop it from the DataFrame if present
    if 'player_id' in test_dataframe.columns:
        player_ids = test_dataframe['player_id']
        test_dataframe = test_dataframe.drop(['player_id'], axis=1)
    else:
        raise ValueError("player_id not found in test_dataframe.")
    
    # Identify numerical columns for adding k-means feature
    numerical_columns = test_dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Add the k-means feature to the test set
    test_dataframe['kmeans_distance'] = kmeans_model.transform(test_dataframe[numerical_columns]).min(axis=1)

    # Use trained base models to generate predictions on the test set
    logreg_probabilities = logreg_pipeline.predict_proba(test_dataframe)[:, 1].reshape(-1, 1)
    svm_probabilities = svm_pipeline.predict_proba(test_dataframe)[:, 1].reshape(-1, 1)

    # Stack the predictions to create new features for the meta-model
    stacked_features_test = np.hstack([logreg_probabilities, svm_probabilities])

    # Predict probabilities on the test set using the meta-model
    ensemble_probabilities = meta_model.predict_proba(stacked_features_test)[:, 1]

    # Create a submission DataFrame
    submission_stacking = pd.DataFrame({'player_id': player_ids, 'drafted': ensemble_probabilities})

    return submission_stacking

