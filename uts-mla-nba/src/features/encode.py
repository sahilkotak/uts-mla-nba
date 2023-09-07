import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_features(dataframe):
    """
    Apply one-hot encoding to the categorical columns and standardize numerical columns in the data.

    Parameters:
    - dataframe: DataFrame, the input data.

    Returns:
    - DataFrame with categorical columns one-hot encoded and numerical columns standardized.
    """
    
    # Identify numerical and categorical columns
    numerical_columns = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = dataframe.select_dtypes(include=['object']).columns.tolist()
    
    # Apply one-hot encoding to categorical columns
    dataframe_encoded = pd.get_dummies(dataframe, columns=categorical_columns, drop_first=True)
    
    # Standardize the numerical columns
    scaler = StandardScaler()
    dataframe_encoded[numerical_columns] = scaler.fit_transform(dataframe[numerical_columns])
    
    return dataframe_encoded