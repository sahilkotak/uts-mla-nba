import re
import pandas as pd
import numpy as np

# Dictionary to map month names to corresponding feet values
month_to_feet = {
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Apr': 4
}


def transform_height_to_inches(height_str):
    """Transform the height string into a numerical format in inches."""
    # List of placeholders and unexpected values in the height column
    INVALID_HEIGHT_VALUES = ["-", "None", "0", "Jr", "So", "Fr"]

    # Convert input to string to ensure consistent handling
    height_str = str(height_str)

    # Immediately return None for any invalid height values
    if height_str in INVALID_HEIGHT_VALUES:
        return None

    # Handle the format like "6'4" which represents 6 feet 4 inches
    if "'" in height_str:
        feet, inches = height_str.split("'")
        return int(feet) * 12 + int(inches)

    # Handle various formats with a dash, such as "X-Jun", "X-Jul", "X-Apr" and "Jun-00", "Jul-00", "Apr-00"
    if "-" in height_str:
        first_part, second_part = height_str.split("-")

        # Handle cases where the first part is a month (e.g., "Jun-00")
        if first_part in month_to_feet:
            return month_to_feet[first_part] * 12 + int(second_part)

        # Handle cases where the second part is the month (e.g., "11-May")
        if second_part in month_to_feet:
            return month_to_feet[second_part] * 12 + int(first_part)

    # Return None for any unhandled formats
    return None


def preprocess_func(df, original_df=None, train_stats=None, missing_threshold=0.01):
    """
    A comprehensive function to preprocess the given dataset based on the steps identified during EDA.

    Parameters:
    - df: DataFrame to be preprocessed.
    - original_df: Original DataFrame before any transformations, needed for certain operations.
    - train_stats: Dictionary with statistics from the training set for imputation on test set.
    - missing_threshold: Percentage threshold below which rows with missing values will be dropped.

    Returns:
    - Preprocessed DataFrame.
    - Dictionary with statistics (useful for test set imputation).
    """

    # Create a copy of the dataframe to ensure no warnings and unexpected behavior
    df = df.copy()

    # 1. Height Transformation
    df['ht'] = df['ht'].apply(transform_height_to_inches)

    # Handle 'yr' column
    year_mapping = {
        'Fr': 1,
        'So': 2,
        'Jr': 3,
        'Sr': 4
    }

    # Check if the column has object (string) type values and apply the mapping
    if df['yr'].dtype == 'O':
        df['yr'] = df['yr'].map(year_mapping)
    
    # Modify the 'Rec_Rank' & 'pick' column
    for col in ['Rec_Rank', 'pick']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 0 if pd.isna(x) else 1)
            
    # Handle missing values in 'dunks_ratio', 'rim_ratio' and 'mid_ratio' based on other relevant columns
    for ratio_col, made_col, made_miss_col in [
        ('dunks_ratio', 'dunksmade', 'dunksmiss_dunksmade')
    ]:
        if ratio_col in df.columns and made_col in df.columns and made_miss_col in df.columns:
            # Calculate missing ratio where possible
            missing_ratio_idx = df[ratio_col].isna() & df[made_col].notna() & df[made_miss_col].notna()
            df.loc[missing_ratio_idx, ratio_col] = df.loc[missing_ratio_idx, made_col] / df.loc[missing_ratio_idx, made_miss_col]
            
            # Handle remaining missing values (if any) in the made and made_miss columns
            for col in [made_col, made_miss_col]:
                df[col].fillna(df[col].mean(), inplace=True)

            # Recalculate any remaining missing ratio values
            df[ratio_col].fillna(df[made_col] / df[made_miss_col], inplace=True)
            
            # # Fill any still remaining missing values in the ratio columns with the mean of existing values
            # df[ratio_col].fillna(df[ratio_col].median(), inplace=True)
    
    # Handle missing values for 'num'
    def clean_num_column(num_val):
        if pd.isna(num_val):
            return num_val
        try:
            return float(num_val)
        except:
            return None

    df['cleaned_num'] = df['num'].apply(clean_num_column)
    
    # Fill NaNs with a placeholder value (-1)
    placeholder_value = -1.0
    df['cleaned_num'].fillna(placeholder_value, inplace=True)
    
    # Drop the original 'num' column
    df.drop(columns=['num'], inplace=True)
    
    # Rename 'cleaned_num' to 'num'
    df.rename(columns={'cleaned_num': 'num'}, inplace=True)
    
    # New step to handle 'ast_tov'
    df['ast_tov'] = df['AST_per'] / df['TO_per']
    
    # Handle division by zero and NaN
    df['ast_tov'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['ast_tov'].fillna(df['ast_tov'].median(), inplace=True)

    # Columns for median and mean imputation based on recommendations
    columns_for_median_imputation = ['rimmade', 'rimmade_rimmiss', 'midmade', 'midmade_midmiss', 'dunksmade',
                                     'dunksmiss_dunksmade', 'dunks_ratio', 'drtg', 'adrtg', 'dporpag', 'bpm', 'obpm',
                                     'dbpm', 'gbpm', 'ogbpm', 'dgbpm', 'oreb', 'dreb', 'treb', 'ast', 'stl', 'blk', 'pts']
    columns_for_mean_imputation = [
        'ht', 'rim_ratio', 'mid_ratio', 'stops', 'mp']

    # If train_stats is not provided (i.e., we're processing the training dataset),
    # calculate and use the statistics of the dataframe being processed
    if not train_stats:
        train_stats = {
            'midmade_midmiss': df['midmade_midmiss'].mean(),
            'dunksmade': df['dunksmade'].mean(),
            'yr_mode': df['yr'].mean(),
            'ht_median': df['ht'].mean(),
            'medians': {column: df[column].median() for column in columns_for_median_imputation},
            'means': {column: df[column].mean() for column in columns_for_mean_imputation},
            'modes': {column: df[column].mode()[0] for column in ['team', 'conf', 'num']}
        }

    # Handle missing values based on the recommendations

    # Median Imputation
    for column in columns_for_median_imputation:
        df[column].fillna(train_stats['medians'][column], inplace=True)

    # Mean Imputation
    for column in columns_for_mean_imputation:
        df[column].fillna(train_stats['means'][column], inplace=True)

    # Mode Imputation for categorical columns
    for column in ['team', 'conf']:
        df[column].fillna(train_stats['modes'][column], inplace=True)

    # Handle 'yr' column missing values after mapping
    df['yr'].fillna(train_stats['yr_mode'], inplace=True)

    # Drop unnecessary columns
    columns_to_drop = ['type', 'year', 'num']

    # Retain 'player_id' for test set
    if not train_stats:
        columns_to_drop.append('player_id')

    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)

    return df, train_stats
