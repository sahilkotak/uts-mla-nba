import re

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

    # Columns for median and mean imputation based on recommendations
    columns_for_median_imputation = ['ast_tov', 'rimmade', 'rimmade_rimmiss', 'midmade', 'midmade_midmiss', 'dunksmade',
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
    for column in ['team', 'conf', 'num']:
        df[column].fillna(train_stats['modes'][column], inplace=True)

    # Handle 'yr' column missing values after mapping
    df['yr'].fillna(train_stats['yr_mode'], inplace=True)

    # Drop unnecessary columns
    columns_to_drop = ['type', 'Rec_Rank', 'pick', 'num']

    # Retain 'player_id' for test set
    if not train_stats:
        columns_to_drop.append('player_id')

    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)

    return df, train_stats
