from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def objective(params):
    try:
        xgb_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                learning_rate=params['learning_rate'],
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                min_child_weight=params['min_child_weight'],
                gamma=params['gamma'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                reg_alpha=params['reg_alpha'],
                use_label_encoder=False,
                eval_metric='logloss'
            ))
        ])
        
        xgb_pipeline.fit(features_train, target_train)
        xgb_probabilities = xgb_pipeline.predict_proba(features_val)[:, 1]
        auroc_score = roc_auc_score(target_val, xgb_probabilities)
        
        if np.isnan(auroc_score):
            print("AUROC is NaN. Skipping...")
            return {'loss': np.inf, 'status': STATUS_OK}
        
        print(f"Current AUROC: {auroc_score}, Best AUROC: {trials.best_trial['result']['loss'] if trials.best_trial else None}")
        return {'loss': -auroc_score, 'status': STATUS_OK}
    except Exception as e:
        print(f"An exception occurred: {e}")
        return {'loss': np.inf, 'status': STATUS_OK}


def train_xgboost_model(train_dataframe, target_column_name, test_size=0.2, random_state=42):
    """
    Train an XGBoost model with hyperparameter tuning using Hyperopt.
    Keep track of the best parameters and print the scores.

    Parameters:
    - train_dataframe: DataFrame with the preprocessed training data.
    - target_column_name: String, name of the target column.
    - test_size: Fraction of the training data to be used as validation set.
    - random_state: Random seed for reproducibility.

    Returns:
    - Trained XGBoost model.
    - AUROC score on the validation set.
    """
    global features_train, features_val, target_train, target_val, preprocessor, trials

    features = train_dataframe.drop(columns=[target_column_name])
    target = train_dataframe[target_column_name]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_size, random_state=random_state)
    
    numerical_columns = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = features.select_dtypes(include=['object']).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', StandardScaler(), numerical_columns),
            ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
        ])

    space = {
        'learning_rate': hp.loguniform('learning_rate', -5, 0),
        'n_estimators': hp.quniform('n_estimators', 50, 500, 1),
        'max_depth': hp.quniform('max_depth', 2, 10, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'gamma': hp.uniform('gamma', 0, 0.5),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'reg_alpha': hp.loguniform('reg_alpha', -5, 2)
    }
    
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials)
    
    best_xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            learning_rate=best['learning_rate'],
            n_estimators=int(best['n_estimators']),
            max_depth=int(best['max_depth']),
            min_child_weight=best['min_child_weight'],
            gamma=best['gamma'],
            subsample=best['subsample'],
            colsample_bytree=best['colsample_bytree'],
            reg_alpha=best['reg_alpha'],
            use_label_encoder=False,
            eval_metric='logloss'
        ))
    ])
    best_xgb_pipeline.fit(features_train, target_train)
    
    xgb_probabilities = best_xgb_pipeline.predict_proba(features_val)[:, 1]
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

    xgb_probabilities = best_xgb_pipeline.predict_proba(test_dataframe)[:, 1]

    submission_xgboost = pd.DataFrame(
        {'player_id': test_dataframe['player_id'], 'drafted': xgb_probabilities})

    return submission_xgboost