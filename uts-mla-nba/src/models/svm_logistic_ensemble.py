from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd


def train_ensemble_model_with_auroc(train_dataframe, target_column_name, test_size=0.2, random_state=42):
    """
    Train an ensemble of Logistic Regression and SVM, and compute AUROC score on a validation set.

    Parameters:
    - train_dataframe: DataFrame with the preprocessed training data.
    - target_column_name: String, name of the target column.
    - test_size: Fraction of the training data to be used as validation set.
    - random_state: Random seed for reproducibility.

    Returns:
    - Trained Logistic Regression and SVM pipelines.
    - AUROC score on the validation set for the ensemble.
    """

    # Split the preprocessed train data into training and validation sets
    features = train_dataframe.drop(columns=[target_column_name])
    target = train_dataframe[target_column_name]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_size, random_state=random_state)

    # Identify numerical and categorical columns
    numerical_columns = features.select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    categorical_columns = features.select_dtypes(
        include=['object']).columns.tolist()

    # Create a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', StandardScaler(), numerical_columns),
            ('categorical', OneHotEncoder(drop='first',
             handle_unknown='ignore'), categorical_columns)
        ])

    # Create pipelines for both logistic regression and SVM
    logreg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    svm_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='linear', probability=True))
    ])

    # Train both models on the training data
    logreg_pipeline.fit(features_train, target_train)
    svm_pipeline.fit(features_train, target_train)

    # Predict probabilities on the validation set using both models
    logreg_probabilities = logreg_pipeline.predict_proba(features_val)[:, 1]
    svm_probabilities = svm_pipeline.predict_proba(features_val)[:, 1]

    # Average the probabilities
    averaged_probabilities = (logreg_probabilities + svm_probabilities) / 2

    # Compute AUROC score
    auroc_score = roc_auc_score(target_val, averaged_probabilities)

    return logreg_pipeline, svm_pipeline, auroc_score


def generate_ensemble_predictions_updated(logreg_model, svm_model, test_dataframe):
    """
    Generate predictions using the trained ensemble models.

    Parameters:
    - logreg_model: Trained Logistic Regression pipeline.
    - svm_model: Trained SVM pipeline.
    - test_dataframe: DataFrame with the test data.

    Returns:
    - Predictions as a DataFrame.
    """

    # Extract the preprocessor from the logistic regression pipeline and update the handle_unknown parameter for OneHotEncoder
    preprocessor_logreg = logreg_model.named_steps['preprocessor']
    for transformer in preprocessor_logreg.transformers_:
        if isinstance(transformer[1], OneHotEncoder):
            transformer[1].set_params(handle_unknown='ignore')

    # Extract the preprocessor from the SVM pipeline and update the handle_unknown parameter for OneHotEncoder
    preprocessor_svm = svm_model.named_steps['preprocessor']
    for transformer in preprocessor_svm.transformers_:
        if isinstance(transformer[1], OneHotEncoder):
            transformer[1].set_params(handle_unknown='ignore')

    # Predict probabilities on the test set using both models
    logreg_probabilities = logreg_model.predict_proba(test_dataframe)[:, 1]
    svm_probabilities = svm_model.predict_proba(test_dataframe)[:, 1]

    # Average the probabilities
    averaged_probabilities = (logreg_probabilities + svm_probabilities) / 2

    # Create a submission DataFrame
    submission_ensemble = pd.DataFrame(
        {'player_id': test_dataframe['player_id'], 'drafted': averaged_probabilities})

    return submission_ensemble
