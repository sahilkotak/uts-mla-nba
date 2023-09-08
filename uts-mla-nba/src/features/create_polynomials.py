from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

def create_polynomial_features(df, target_col=None, degree=2, interaction_only=False, include_bias=False):
    """
    Create polynomial features from a preprocessed dataframe.
    
    Parameters:
    - df: Pandas DataFrame containing the original features.
    - degree: Integer, degree of the polynomial features. Default is 2.
    - interaction_only: Boolean, whether to include only interaction features. Default is False.
    - include_bias: Boolean, whether to include a bias column in the output. Default is False.
    
    Returns:
    - Pandas DataFrame containing the polynomial features.
    """
    # If target column specified, separate it from features
    if target_col:
        y = df[target_col]
        df = df.drop(columns=[target_col])
    else:
        y = None
    
    # Automatically detect numerical columns based on dtype
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Separate numerical and categorical columns
    df_numerical = df[numerical_cols]
    df_categorical = df.drop(columns=numerical_cols)
    
    # Create polynomial features for numerical columns
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    poly_features = poly.fit_transform(df_numerical)
    
    # Get feature names
    feature_names = poly.get_feature_names_out(input_features=numerical_cols)
    
    # Create a new dataframe for polynomial features
    df_poly = pd.DataFrame(poly_features, columns=feature_names)
    
    # Concatenate with original categorical features
    df_combined = pd.concat([df_poly, df_categorical.reset_index(drop=True)], axis=1)
    
    # Reattach target column if it was specified
    if y is not None:
        df_combined[target_col] = y
    
    return df_combined