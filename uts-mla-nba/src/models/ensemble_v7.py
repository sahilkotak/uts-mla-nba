# Changelog:
# Prev Ver: v6
# Added K-NN and remove Logistic and SVM

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer
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

def train_stacking_ensemble(train_dataframe, target_column_name, n_folds=5, random_state=42):
    features = train_dataframe.drop(columns=[target_column_name])
    target = train_dataframe[target_column_name]
    
    numerical_columns = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = features.select_dtypes(include=['object']).columns.tolist()
    
    n_clusters = find_optimal_clusters(features[numerical_columns])
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=0).fit(features[numerical_columns])
    features['kmeans_distance'] = kmeans_model.transform(features[numerical_columns]).min(axis=1)
    numerical_columns.append('kmeans_distance')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', StandardScaler(), numerical_columns),
            ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
        ])

    knn_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier())
    ])

    knn_param_grid = {
        'classifier__n_neighbors': [3, 5, 7, 9, 11],
        'classifier__weights': ['uniform', 'distance'],
    }

    stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    grid_search_knn = GridSearchCV(knn_pipeline, knn_param_grid, cv=stratified_kfold, scoring=make_scorer(roc_auc_score))
    grid_search_knn.fit(features, target)

    auroc_mean_knn = grid_search_knn.cv_results_['mean_test_score'].max()

    return grid_search_knn.best_estimator_, auroc_mean_knn, kmeans_model

def generate_stacking_predictions(knn_pipeline, kmeans_model, test_dataframe):
    if 'player_id' in test_dataframe.columns:
        player_ids = test_dataframe['player_id']
        test_dataframe = test_dataframe.drop(['player_id'], axis=1)
    else:
        raise ValueError("player_id not found in test_dataframe.")
    
    numerical_columns = test_dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    test_dataframe['kmeans_distance'] = kmeans_model.transform(test_dataframe[numerical_columns]).min(axis=1)

    knn_probabilities = knn_pipeline.predict_proba(test_dataframe)[:, 1]
    submission_stacking = pd.DataFrame({'player_id': player_ids, 'drafted': knn_probabilities})

    return submission_stacking
