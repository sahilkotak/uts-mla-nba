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
        {'player_id': test_dataframe['player_id'], 'drafted': ensemble_probabilities})  # Assuming 'player_id' is the index

    return submission_stacking