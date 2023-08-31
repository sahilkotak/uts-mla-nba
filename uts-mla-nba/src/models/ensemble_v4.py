from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
import numpy as np
import pandas as pd

def stacking_ensemble(train_dataframe, target_column_name, test_size=0.2, random_state=42):
    """
    Train a complete stacking ensemble of Logistic Regression, SVM, XGBoost, Random Forest, 
    and compute AUROC score on a validation set.

    Parameters:
    - train_dataframe: DataFrame with the preprocessed training data.
    - target_column_name: String, name of the target column.
    - test_size: Fraction of the training data to be used as validation set.
    - random_state: Random seed for reproducibility.

    Returns:
    - Trained models as a dictionary.
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
    
    # Create a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', StandardScaler(), numerical_columns),
            ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
        ])

    # Initialize models
    logistic_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    svm_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='linear', probability=True))
    ])
    
    xgboost_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state))
    ])
    
    random_forest_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=random_state))
    ])
    
    # List of base models
    base_models = [
        ('logistic_model', logistic_model),
        ('svm_model', svm_model),
        ('xgboost_model', xgboost_model),
        ('random_forest_model', random_forest_model)
    ]
    
    # Initialize and train Stacking Classifier
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=5)
    stacking_model.fit(features_train, target_train)
    
    # Initialize and train Voting Classifier
    voting_model = VotingClassifier(estimators=base_models, voting='soft')
    voting_model.fit(features_train, target_train)
    
    # Evaluate the Stacking model
    stacking_probabilities = stacking_model.predict_proba(features_val)[:, 1]
    stacking_auroc_score = roc_auc_score(target_val, stacking_probabilities)
    
    # Evaluate the Voting model
    voting_probabilities = voting_model.predict_proba(features_val)[:, 1]
    voting_auroc_score = roc_auc_score(target_val, voting_probabilities)
    
    # Store trained models and their scores
    trained_models = {
        'Logistic Regression': logistic_model,
        'SVM': svm_model,
        'XGBoost': xgboost_model,
        'Random Forest': random_forest_model,
        'Stacking Classifier': stacking_model,
        'Voting Classifier': voting_model,
    }
    
    auroc_scores = {
        'Stacking Classifier AUROC': stacking_auroc_score,
        'Voting Classifier AUROC': voting_auroc_score,
    }
    
    return trained_models, auroc_scores
