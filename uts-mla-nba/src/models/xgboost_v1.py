from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def train_xgboost_model(train_dataframe, target_column_name, test_size=0.2, random_state=42):
    """
    Train an XGBoost model and compute AUROC score on a validation set.
    Hyperparameter tuning is done using GridSearchCV.

    Parameters:
    - train_dataframe: DataFrame with the preprocessed training data.
    - target_column_name: String, name of the target column.
    - test_size: Fraction of the training data to be used as validation set.
    - random_state: Random seed for reproducibility.

    Returns:
    - Trained XGBoost model.
    - AUROC score on the validation set.
    """

    # Split the preprocessed train data into training and validation sets
    features = train_dataframe.drop(columns=[target_column_name])
    target = train_dataframe[target_column_name]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_size, random_state=random_state)

    # Identify numerical and categorical columns
    numerical_columns = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = features.select_dtypes(include=['object']).columns.tolist()

    # Create a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', StandardScaler(), numerical_columns),
            ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
        ])

    # Create pipeline for XGBoost
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])

    # Define hyperparameter grid
    param_grid = {
        'classifier__learning_rate': [0.01, 0.1, 0.3],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 7]
    }

    # Create GridSearchCV object
    grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)

    # Perform hyperparameter search
    grid_search.fit(features_train, target_train)

    # Get the best estimator
    best_xgb_pipeline = grid_search.best_estimator_

    # Predict probabilities on the validation set
    xgb_probabilities = best_xgb_pipeline.predict_proba(features_val)[:, 1]

    # Compute AUROC score
    auroc_score = roc_auc_score(target_val, xgb_probabilities)

    return best_xgb_pipeline, auroc_score

def generate_xgboost_predictions(best_xgb_pipeline, test_dataframe):
    """
    Generate predictions using the trained XGBoost model.

    Parameters:
    - best_xgb_pipeline: Trained XGBoost pipeline.
    - test_dataframe: DataFrame with the test data.

    Returns:
    - Predictions as a DataFrame.
    """

    # Predict probabilities on the test set
    xgb_probabilities = best_xgb_pipeline.predict_proba(test_dataframe)[:, 1]

    # Create a submission DataFrame
    submission_xgboost = pd.DataFrame(
        {'player_id': test_dataframe['player_id'], 'drafted': xgb_probabilities})

    return submission_xgboost