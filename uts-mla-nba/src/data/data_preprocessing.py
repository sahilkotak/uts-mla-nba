from data.data_preparation import transform_height_to_inches


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

    # 2. Handle Missing Values

    # If train_stats is not provided (i.e., we're processing the training dataset),
    # calculate and use the statistics of the dataframe being processed
    if not train_stats:
        train_stats = {
            'midmade_midmiss': df['midmade_midmiss'].median(),
            'dunksmade': df['dunksmade'].median(),
            'yr_mode': df['yr'].mode()[0],
            'medians': {column: df[column].median() for column in ['midmade', 'rimmade', 'mid_ratio',
                                                                   'dunksmiss_dunksmade', 'rim_ratio',
                                                                   'rimmade_rimmiss', 'dunks_ratio', 'ast_tov']}
        }

    df['midmade_midmiss'].fillna(train_stats['midmade_midmiss'], inplace=True)
    df['dunksmade'].fillna(train_stats['dunksmade'], inplace=True)

    # Handle 'yr' column
    year_mapping = {
        'Fr': 1,
        'So': 2,
        'Jr': 3,
        'Sr': 4
    }
    df['yr'] = df['yr'].map(year_mapping)
    df['yr'].fillna(train_stats['yr_mode'], inplace=True)

    # Handle columns with moderate missing values
    for column, median_value in train_stats['medians'].items():
        df[column].fillna(median_value, inplace=True)

    # Drop rows with a small percentage of missing values
    missing_data_percentage = df.isnull().sum() / len(df)
    columns_to_check_for_missing = missing_data_percentage[missing_data_percentage < missing_threshold].index.tolist(
    )
    df = df.dropna(subset=columns_to_check_for_missing)

    # 3. Drop Columns and Rows

    # Drop 'num' column
    if 'num' in df.columns:
        df = df.drop('num', axis=1)

    # Drop rows that originally had NaN values in the 'ht' column
    if original_df is not None:
        df = df[~original_df['ht'].isnull()]

    # Drop 'pick' column if it exists
    if 'pick' in df.columns:
        df = df.drop('pick', axis=1)

    # Drop rows with NaN values in certain columns
    columns_to_check = ['ht', 'yr', 'oreb', 'dreb',
                        'ast', 'treb', 'stl', 'pts', 'blk', 'mp']
    df = df.dropna(subset=columns_to_check)

    # Drop 'Rec_Rank' column if it exists
    if 'Rec_Rank' in df.columns:
        df = df.drop('Rec_Rank', axis=1)

    return df, train_stats
