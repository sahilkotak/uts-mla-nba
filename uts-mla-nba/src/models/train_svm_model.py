from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def train_svm_model(train_dataframe, target_column_name, test_size=0.2, random_state=42):
    """
    Train a Linear SVM model and compute AUROC score on a validation set.

    Parameters:
    - train_dataframe: DataFrame with the preprocessed training data.
    - target_column_name: String, name of the target column.
    - test_size: Fraction of the training data to be used as validation set.
    - random_state: Random seed for reproducibility.

    Returns:
    - Trained SVM pipeline.
    - AUROC score on the validation set.
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

    # Create a pipeline with the preprocessor and SVM
    svm_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='linear', probability=True))
    ])

    # Train the SVM on the training data
    svm_pipeline.fit(features_train, target_train)

    # Predict probabilities on the validation set
    val_probabilities = svm_pipeline.predict_proba(features_val)[:, 1]

    # Compute AUROC score
    auroc_score = roc_auc_score(target_val, val_probabilities)

    return svm_pipeline, auroc_score
