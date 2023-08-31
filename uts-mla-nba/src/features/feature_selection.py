from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

def select_important_features(preprocessed_data, target_column='drafted', top_n_numerical=20, top_n_categorical=5):
    """
    Select the top N most important numerical and categorical features based on feature importances and chi-square test.
    
    Parameters:
    - preprocessed_data: DataFrame containing the preprocessed data.
    - target_column: The name of the target variable column.
    - top_n_numerical: The number of top numerical features to select.
    - top_n_categorical: The number of top categorical features to select.
    
    Returns:
    - DataFrame containing only the selected top features.
    """
    
    # Drop 'player_id' if exists
    features = preprocessed_data.drop(columns=[target_column, 'player_id'], errors='ignore')
    target = preprocessed_data[target_column]
    
    # Identify numerical and categorical columns
    numerical_columns = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = features.select_dtypes(include=['object']).columns.tolist()
    
    # Select top N numerical features based on feature importances
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(features[numerical_columns], target)
    numerical_importances = rf_classifier.feature_importances_
    numerical_importance_df = pd.DataFrame({
        'Feature': numerical_columns,
        'Importance': numerical_importances
    })
    top_numerical_features = numerical_importance_df.sort_values(by='Importance', ascending=False).iloc[:top_n_numerical]['Feature'].tolist()
    
    # Select top N categorical features based on chi-square test
    # Encode the categorical variables to make them suitable for chi-square test
    categorical_data = pd.get_dummies(features[categorical_columns])
    chi_selector = SelectKBest(chi2, k=top_n_categorical)
    chi_selector.fit(categorical_data, target)
    top_categorical_features_encoded = [categorical_data.columns[i] for i in chi_selector.get_support(indices=True)]
    
    # Map back to original categorical columns
    top_categorical_features = list(set([col.split('_')[0] for col in top_categorical_features_encoded]))
    
    # Create a new DataFrame with only the selected top features
    selected_top_features_data = preprocessed_data[top_numerical_features + top_categorical_features + [target_column]]
    
    return selected_top_features_data