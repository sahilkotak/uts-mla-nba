from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

# Initialize a global variable for the scaler, to be set when training data is scaled
global_scaler = None

global_scaler = None

def scale_data(dataframe, target_column_name, is_train):
    global global_scaler
    
    # Select only the numerical columns
    numerical_columns = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove the target column from the list of numerical columns
    if target_column_name in numerical_columns:
        numerical_columns.remove(target_column_name)
    
    if is_train:
        global_scaler = StandardScaler()
        dataframe[numerical_columns] = global_scaler.fit_transform(dataframe[numerical_columns])
    else:
        if global_scaler is not None:
            dataframe[numerical_columns] = global_scaler.transform(dataframe[numerical_columns])
        else:
            raise ValueError("The scaler has not been fitted yet. Please run the function on the training data first.")
    
    return dataframe

def find_optimal_clusters(data, max_clusters=10):
    """
    Finds the optimal number of clusters using silhouette score.
    """
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k).fit(data)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append((k, score))
        
    # Choose the number of clusters that gives the maximum silhouette score
    optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
    return optimal_k

def add_kmeans_feature(dataframe, target_column_name, n_clusters=None):
    # Select only the numerical columns
    numerical_columns = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove the target column from the list of numerical columns
    if target_column_name in numerical_columns:
        numerical_columns.remove(target_column_name)
    
    # Determine the optimum number of clusters if not provided
    if n_clusters is None:
        n_clusters = find_optimal_clusters(dataframe[numerical_columns])
    
    # Fit the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(dataframe[numerical_columns])
    
    # Calculate the distance to cluster center for each sample and add as a feature
    distances = kmeans.transform(dataframe[numerical_columns])
    min_distances = np.min(distances, axis=1)
    dataframe['kmeans_distance'] = min_distances
    
    return dataframe
