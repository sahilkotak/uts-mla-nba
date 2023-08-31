from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

# Recreate the function to remove the least important features
def remove_least_important_features(preprocessed_data, target_column='drafted', bottom_n_numerical=5, bottom_n_categorical=5):
    """
    Remove the bottom N least important numerical and categorical features based on feature importances and chi-square test.
    
    Parameters:
    - preprocessed_data: DataFrame containing the preprocessed data.
    - target_column: The name of the target variable column.
    - bottom_n_numerical: The number of bottom numerical features to remove.
    - bottom_n_categorical: The number of bottom categorical features to remove.
    
    Returns:
    - DataFrame with the least important features removed.
    """
    
    # Drop 'player_id' if exists
    features = preprocessed_data.drop(columns=[target_column, 'player_id'], errors='ignore')
    target = preprocessed_data[target_column]
    
    # Identify numerical and categorical columns
    numerical_columns = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = features.select_dtypes(include=['object']).columns.tolist()
    
    # Select bottom N numerical features based on feature importances
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(features[numerical_columns], target)
    numerical_importances = rf_classifier.feature_importances_
    numerical_importance_df = pd.DataFrame({
        'Feature': numerical_columns,
        'Importance': numerical_importances
    })
    bottom_numerical_features = numerical_importance_df.sort_values(by='Importance', ascending=True).iloc[:bottom_n_numerical]['Feature'].tolist()
    
    # Select bottom N categorical features based on chi-square test
    categorical_data = pd.get_dummies(features[categorical_columns])
    chi_selector = SelectKBest(chi2, k='all')
    chi_selector.fit(categorical_data, target)
    categorical_scores = chi_selector.scores_
    categorical_importance_df = pd.DataFrame({
        'Feature': categorical_data.columns,
        'Score': categorical_scores
    })
    bottom_categorical_features_encoded = categorical_importance_df.sort_values(by='Score', ascending=True).iloc[:bottom_n_categorical]['Feature'].tolist()
    
    # Map back to original categorical columns
    bottom_categorical_features = list(set([col.split('_')[0] for col in bottom_categorical_features_encoded]))
    
    # Create a new DataFrame with only the selected top features
    selected_features_data = preprocessed_data.drop(columns=bottom_numerical_features + bottom_categorical_features, errors='ignore')
    
    return selected_features_data