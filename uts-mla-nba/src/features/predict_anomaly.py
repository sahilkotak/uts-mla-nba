from sklearn.ensemble import IsolationForest

def calculate_anomaly_scores(df, columns_to_exclude, contamination=0.05, random_state=42):
    """
    Calculate anomaly scores using Isolation Forest algorithm.

    Parameters:
    - df: DataFrame, the data to calculate anomaly scores for.
    - columns_to_exclude: List of columns to exclude for anomaly score calculation.
    - contamination: The amount of contamination of the data set, i.e., the proportion of outliers in the data set. 
                    Used when fitting IsolationForest. Range is (0, 0.5).
    - random_state: Random seed for reproducibility.

    Returns:
    - DataFrame with an additional column 'anomaly_score'.
    """
    # Select columns to use by removing the excluded columns
    columns_to_use = [col for col in df.columns if col not in columns_to_exclude]
    
    isolation_forest = IsolationForest(contamination=contamination, random_state=random_state)
    isolation_forest.fit(df[columns_to_use])
    
    anomaly_scores = isolation_forest.decision_function(df[columns_to_use])
    
    df['anomaly_score'] = anomaly_scores
    
    return df