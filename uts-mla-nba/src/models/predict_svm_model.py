import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def generate_svm_predictions(svm_model, test_dataframe):
    """
    Generate predictions using the trained SVM model.

    Parameters:
    - svm_model: Trained SVM pipeline.
    - test_dataframe: DataFrame with the test data.

    Returns:
    - Predictions as a DataFrame.
    """

    # Extract the preprocessor from the SVM pipeline and update the handle_unknown parameter for OneHotEncoder
    preprocessor = svm_model.named_steps['preprocessor']
    for transformer in preprocessor.transformers_:
        if isinstance(transformer[1], OneHotEncoder):
            transformer[1].set_params(handle_unknown='ignore')

    # Predict probabilities on the test set
    test_probabilities = svm_model.predict_proba(test_dataframe)[:, 1]

    # Create a submission DataFrame
    submission_svm = pd.DataFrame(
        {'player_id': test_dataframe['player_id'], 'drafted': test_probabilities})

    return submission_svm
