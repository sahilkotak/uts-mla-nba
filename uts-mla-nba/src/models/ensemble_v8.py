# Changelog:
# Prev Ver: v3
# Added Outlier Detection using One-Class SVM
# Added SGD model in the pipeline

from sklearn.svm import OneClassSVM, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

def train_stacking_ensemble(train_dataframe, target_column_name, test_size=0.2, random_state=42):
    """
    Train a stacking ensemble of Logistic Regression, SVM, and SGD, with One-Class SVM for outlier detection,
    and compute AUROC score on a validation set.

    Parameters:
    - train_dataframe: DataFrame with the preprocessed training data.
    - target_column_name: String, name of the target column.
    - test_size: Fraction of the training data to be used as validation set.
    - random_state: Random seed for reproducibility.

    Returns:
    - Trained Logistic Regression, SVM, SGD, and meta-model pipelines.
    - AUROC score on the validation set for the stacking ensemble.
    - Trained One-Class SVM for outlier detection.
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

    # Apply preprocessing to the training data for outlier detection
    preprocessed_features_train = preprocessor.fit_transform(features_train)
    
    # Outlier detection using One-Class SVM
    outlier_detector = OneClassSVM(kernel='linear', nu=0.1)
    outlier_detector.fit(preprocessed_features_train)
    outlier_flags_train = outlier_detector.predict(preprocessed_features_train)

    # Filter out the outliers
    inlier_indices_train = np.where(outlier_flags_train == 1)[0]
    inlier_features_train = features_train.iloc[inlier_indices_train]
    inlier_target_train = target_train.iloc[inlier_indices_train]
    
    # Create pipelines for logistic regression, SVM, and SGD
    logreg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    svm_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='linear', probability=True))
    ])

    sgd_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SGDClassifier(loss='log', max_iter=1000))
    ])
    
    # Train models on the inlier training data
    logreg_pipeline.fit(inlier_features_train, inlier_target_train)
    svm_pipeline.fit(inlier_features_train, inlier_target_train)
    sgd_pipeline.fit(inlier_features_train, inlier_target_train)

    # Use trained models to generate predictions on the validation set
    logreg_probabilities = logreg_pipeline.predict_proba(features_val)[:, 1].reshape(-1, 1)
    svm_probabilities = svm_pipeline.predict_proba(features_val)[:, 1].reshape(-1, 1)
    sgd_probabilities = sgd_pipeline.predict_proba(features_val)[:, 1].reshape(-1, 1)

    # Stack the predictions to create new features for the meta-model
    stacked_features_val = np.hstack([logreg_probabilities, svm_probabilities, sgd_probabilities])

    # Train the meta-model on the stacked predictions
    meta_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    meta_model.fit(stacked_features_val, target_val)

    # Predict probabilities on the validation set using the meta-model
    ensemble_probabilities = meta_model.predict_proba(stacked_features_val)[:, 1]

    # Compute AUROC score
    auroc_score = roc_auc_score(target_val, ensemble_probabilities)

    return logreg_pipeline, svm_pipeline, sgd_pipeline, meta_model, auroc_score, outlier_detector


def generate_stacking_predictions(logreg_pipeline, svm_pipeline, sgd_pipeline, meta_model, outlier_detector, test_dataframe):
    """
    Generate predictions using a stacking ensemble for a given test set.

    Parameters:
    - test_dataframe: DataFrame containing the features for the test set.
    - logreg_pipeline: Trained pipeline for logistic regression.
    - svm_pipeline: Trained pipeline for support vector machines.
    - sgd_pipeline: Trained pipeline for stochastic gradient descent.
    - meta_model: Trained meta-model.
    - outlier_detector: Trained One-Class SVM for outlier detection.

    Returns:
    - DataFrame containing the ensemble's predicted probabilities for the test set.
    """

    # Apply outlier detection
    preprocessed_features_test = logreg_pipeline.named_steps['preprocessor'].transform(test_dataframe)
    outlier_flags_test = outlier_detector.predict(preprocessed_features_test)

    # Filter out the outliers from the test set
    inlier_indices_test = np.where(outlier_flags_test == 1)[0]
    inlier_features_test = test_dataframe.iloc[inlier_indices_test]

    # Generate predictions for the inlier test set using the base classifiers
    logreg_probabilities = logreg_pipeline.predict_proba(inlier_features_test)[:, 1].reshape(-1, 1)
    svm_probabilities = svm_pipeline.predict_proba(inlier_features_test)[:, 1].reshape(-1, 1)
    sgd_probabilities = sgd_pipeline.predict_proba(inlier_features_test)[:, 1].reshape(-1, 1)

    # Stack the predictions to create new features for the meta-model
    stacked_features_test = np.hstack([logreg_probabilities, svm_probabilities, sgd_probabilities])

    # Generate predictions for the inlier test set using the meta-model
    ensemble_probabilities = meta_model.predict_proba(stacked_features_test)[:, 1]

    # Create a DataFrame to store the ensemble's predicted probabilities
    prediction_dataframe = pd.DataFrame({
        'player_id': test_dataframe.iloc[inlier_indices_test].index,
        'predicted_probability': ensemble_probabilities
    })

    return prediction_dataframe