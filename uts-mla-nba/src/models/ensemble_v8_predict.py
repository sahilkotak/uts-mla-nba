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

# def generate_stacking_predictions(logreg_pipeline, svm_pipeline, sgd_pipeline, meta_model, outlier_detector, test_dataframe):
#     """
#     Generate predictions using a stacking ensemble for a given test set.
    
#     Parameters:
#     - logreg_pipeline: Trained pipeline for logistic regression.
#     - svm_pipeline: Trained pipeline for support vector machines.
#     - sgd_pipeline: Trained pipeline for stochastic gradient descent.
#     - meta_model: Trained meta-model.
#     - outlier_detector: Trained One-Class SVM for outlier detection.
#     - test_dataframe: DataFrame containing the features for the test set.

#     Returns:
#     - DataFrame containing the ensemble's predicted probabilities for the test set.
#     """

#     # Separate player_id for later use
#     player_ids = test_dataframe.index if 'player_id' not in test_dataframe.columns else test_dataframe['player_id']

#     # Make sure player_id is not used in prediction
#     features_test = test_dataframe.drop(['player_id'], axis=1, errors='ignore')

#     # Apply outlier detection
#     preprocessed_features_test = logreg_pipeline.named_steps['preprocessor'].transform(features_test)
    
#     # Make sure outlier_detector is expecting the same number of features
#     if preprocessed_features_test.shape[1] != outlier_detector.n_features_in_:
#         raise ValueError(f"Feature mismatch: Outlier detector expects {outlier_detector.n_features_in_} features but got {preprocessed_features_test.shape[1]}")
    
#     outlier_flags_test = outlier_detector.predict(preprocessed_features_test)

#     # Filter out the outliers from the test set
#     inlier_indices_test = np.where(outlier_flags_test == 1)[0]
#     inlier_features_test = features_test.iloc[inlier_indices_test]

#     # Generate predictions for the inlier test set using the base classifiers
#     logreg_probabilities = logreg_pipeline.predict_proba(inlier_features_test)[:, 1].reshape(-1, 1)
#     svm_probabilities = svm_pipeline.predict_proba(inlier_features_test)[:, 1].reshape(-1, 1)
#     sgd_probabilities = sgd_pipeline.predict_proba(inlier_features_test)[:, 1].reshape(-1, 1)

#     # Stack the predictions to create new features for the meta-model
#     stacked_features_test = np.hstack([logreg_probabilities, svm_probabilities, sgd_probabilities])

#     # Generate predictions for the inlier test set using the meta-model
#     ensemble_probabilities = meta_model.predict_proba(stacked_features_test)[:, 1]

#     # Create a DataFrame to store the ensemble's predicted probabilities
#     prediction_dataframe = pd.DataFrame({
#         'player_id': player_ids.iloc[inlier_indices_test],
#         'predicted_probability': ensemble_probabilities
#     })

#     return prediction_dataframe

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

    # Use the same preprocessor that was used in the training phase for outlier detection
    preprocessor = logreg_pipeline.named_steps['preprocessor']
    preprocessed_features_test = preprocessor.transform(test_dataframe)
    
    # Make sure outlier_detector is expecting the same number of features
    if preprocessed_features_test.shape[1] != outlier_detector.n_features_in_:
        raise ValueError(f"Feature mismatch: Outlier detector expects {outlier_detector.n_features_in_} features but got {preprocessed_features_test.shape[1]}")

    # Apply outlier detection
    outlier_flags_test = outlier_detector.predict(preprocessed_features_test)

    
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