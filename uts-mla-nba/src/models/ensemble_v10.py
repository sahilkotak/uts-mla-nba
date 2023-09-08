from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
import pandas as pd

# Objective function for hyperparameter optimization
def objective(params, classifier_type, features_train, target_train, features_val, target_val, preprocessor, smote, trials):
    try:
        if classifier_type == 'logreg':
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', smote),
                ('classifier', LogisticRegression(C=params['C'], max_iter=5000))
            ])
        elif classifier_type == 'svm':
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', smote),
                ('classifier', SVC(C=params['C'], kernel='linear', probability=True))
            ])
        
        pipeline.fit(features_train, target_train)
        probabilities = pipeline.predict_proba(features_val)[:, 1]
        auroc_score = roc_auc_score(target_val, probabilities)
        
        if np.isnan(auroc_score):
            print("AUROC is NaN. Skipping...")
            return {'loss': np.inf, 'status': STATUS_OK}
        
        print(f"Current AUROC: {auroc_score}, Best AUROC: {trials.best_trial['result']['loss'] if trials.best_trial else None}")
        return {'loss': -auroc_score, 'status': STATUS_OK}
    except Exception as e:
        print(f"An exception occurred: {e}")
        return {'loss': np.inf, 'status': STATUS_OK}


def train_stacking_ensemble(train_dataframe, target_column_name, test_size=0.2, random_state=42):
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
    
    # Apply SMOTE to the training set
    smote = SMOTE(random_state=random_state)
    
    # Define hyperparameter spaces
    space_logreg = {'C': hp.loguniform('C_logreg', -5, 2)}
    space_svm = {'C': hp.loguniform('C_svm', -5, 2)}
    
    # Optimize Logistic Regression
    trials_logreg = Trials()
    best_logreg = fmin(fn=lambda params: objective(params, 'logreg', features_train, target_train, features_val, target_val, preprocessor, smote, trials_logreg),
                   space=space_logreg,
                   algo=tpe.suggest,
                   max_evals=100,
                   trials=trials_logreg)
    
    # Optimize SVM
    trials_svm = Trials()
    best_svm = fmin(fn=lambda params: objective(params, 'svm', features_train, target_train, features_val, target_val, preprocessor, smote, trials_svm),
                space=space_svm,
                algo=tpe.suggest,
                max_evals=50,
                trials=trials_svm)
    
    # Train models with best parameters using the SMOTE-augmented training set
    logreg_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', smote),
        ('classifier', LogisticRegression(C=best_logreg['C_logreg'], max_iter=5000))
    ])
    logreg_pipeline.fit(features_train, target_train)
    
    svm_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', smote),
        ('classifier', SVC(C=best_svm['C_svm'], kernel='linear', probability=True))
    ])
    svm_pipeline.fit(features_train, target_train)
    
    # Use trained models to generate predictions on the validation set
    logreg_probabilities = logreg_pipeline.predict_proba(features_val)[:, 1].reshape(-1, 1)
    svm_probabilities = svm_pipeline.predict_proba(features_val)[:, 1].reshape(-1, 1)

    # Stack the predictions to create new features for the meta-model
    stacked_features_val = np.hstack([logreg_probabilities, svm_probabilities])

    # Train the meta-model on the stacked predictions
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(stacked_features_val, target_val)

    # Predict probabilities on the validation set using the meta-model
    ensemble_probabilities = meta_model.predict_proba(stacked_features_val)[:, 1]

    # Compute AUROC score
    auroc_score = roc_auc_score(target_val, ensemble_probabilities)

    return logreg_pipeline, svm_pipeline, meta_model, auroc_score



def generate_stacking_predictions(logreg_pipeline, svm_pipeline, meta_model, test_dataframe):
    """
    Generate predictions using the trained stacking ensemble.

    Parameters:
    - logreg_pipeline: Trained Logistic Regression pipeline.
    - svm_pipeline: Trained SVM pipeline.
    - meta_model: Trained meta-model for stacking.
    - test_dataframe: DataFrame with the test data.

    Returns:
    - Predictions as a DataFrame.
    """

    # Use trained base models to generate predictions on the test set
    logreg_probabilities = logreg_pipeline.predict_proba(test_dataframe)[
        :, 1].reshape(-1, 1)
    svm_probabilities = svm_pipeline.predict_proba(test_dataframe)[
        :, 1].reshape(-1, 1)

    # Stack the predictions to create new features for the meta-model
    stacked_features_test = np.hstack(
        [logreg_probabilities, svm_probabilities])

    # Predict probabilities on the test set using the meta-model
    ensemble_probabilities = meta_model.predict_proba(
        stacked_features_test)[:, 1]

    # Create a submission DataFrame
    submission_stacking = pd.DataFrame(
        {'player_id': test_dataframe['player_id'], 'drafted': ensemble_probabilities})

    return submission_stacking
