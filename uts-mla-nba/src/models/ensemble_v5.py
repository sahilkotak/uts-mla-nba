####### Continuation from ensemble_v3.py ############

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

def train_stacking_ensemble(train_dataframe, target_column_name, test_size=0.2, random_state=42):
    # Split the preprocessed train data into training and validation sets
    features = train_dataframe.drop(columns=[target_column_name])
    target = train_dataframe[target_column_name]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target)

    # Identify numerical and categorical columns
    numerical_columns = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = features.select_dtypes(include=['object']).columns.tolist()

    # Create a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', StandardScaler(), numerical_columns),
            ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
        ])

    # Create pipelines for Logistic Regression, ElasticNet Logistic Regression, and Linear SVM
    logreg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    
    elasticnet_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, penalty='elasticnet', solver='saga', l1_ratio=0.5, class_weight='balanced'))
    ])

    svm_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='linear', probability=True, class_weight='balanced'))
    ])
    
    # Define hyperparameters and their possible values for GridSearch
    param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    
    # GridSearch for each pipeline
    logreg_gs = GridSearchCV(logreg_pipeline, param_grid, cv=5, scoring='roc_auc')
    elasticnet_gs = GridSearchCV(elasticnet_pipeline, param_grid, cv=5, scoring='roc_auc')
    svm_gs = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring='roc_auc')
    
    # Train the models using GridSearch
    logreg_gs.fit(features_train, target_train)
    elasticnet_gs.fit(features_train, target_train)
    svm_gs.fit(features_train, target_train)
    
    # Extract the best pipelines from GridSearch
    best_logreg_pipeline = logreg_gs.best_estimator_
    best_elasticnet_pipeline = elasticnet_gs.best_estimator_
    best_svm_pipeline = svm_gs.best_estimator_
    
    # Create a Stacking Classifier
    estimators = [
        ('best_logreg', best_logreg_pipeline),
        ('best_elasticnet', best_elasticnet_pipeline),
        ('best_svm', best_svm_pipeline)
    ]
    stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    
    # Train the stacking classifier
    stacking_classifier.fit(features_train, target_train)
    
    # Evaluate the model
    auroc_score = roc_auc_score(target_val, stacking_classifier.predict_proba(features_val)[:, 1])

    return best_logreg_pipeline, best_elasticnet_pipeline, best_svm_pipeline, stacking_classifier, auroc_score



import pandas as pd

def generate_stacking_predictions(stacking_classifier, test_dataframe):
    """
    Generate predictions using the trained stacking ensemble.

    Parameters:
    - stacking_classifier: Trained StackingClassifier for stacking.
    - test_dataframe: DataFrame with the test data.

    Returns:
    - Predictions as a DataFrame.
    """

    # Predict probabilities on the test set using the Stacking Classifier
    ensemble_probabilities = stacking_classifier.predict_proba(test_dataframe)[:, 1]

    # Create a submission DataFrame
    submission_stacking = pd.DataFrame(
        {'player_id': test_dataframe.index, 'drafted': ensemble_probabilities})  # Assuming 'player_id' is the index

    return submission_stacking
