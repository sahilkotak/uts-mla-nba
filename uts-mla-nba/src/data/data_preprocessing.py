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
            'ht_median': df['ht'].median(),
            'medians': {column: df[column].median() for column in ['midmade', 'rimmade', 'mid_ratio',
                                                                   'dunksmiss_dunksmade', 'rim_ratio',
                                                                   'rimmade_rimmiss', 'dunks_ratio', 'ast_tov',
                                                                   'drtg', 'adrtg', 'dporpag', 'stops', 'bpm', 
                                                                   'obpm', 'dbpm', 'gbpm', 'ogbpm', 'dgbpm']}
        }

    #TO VERIFY: Instead of dropping rows with NaN in 'ht', we impute them with the median height from the training dataset.
    df['ht'].fillna(train_stats['ht_median'], inplace=True)

    df['midmade_midmiss'].fillna(train_stats['midmade_midmiss'], inplace=True)
    df['dunksmade'].fillna(train_stats['dunksmade'], inplace=True)

    #TO VERIFY: Instead of dropping rows that don't match our expected values (Fr, So, Jr, Sr), we impute them with the mode (most frequent value) of the training dataset.
    df['yr'].fillna(train_stats['yr_mode'], inplace=True)

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

    # Handle columns with moderate missing values
    for column, median_value in train_stats['medians'].items():
        df[column].fillna(median_value, inplace=True)

    # 3. Drop Columns and Rows

    # Drop 'num' column
    if 'num' in df.columns:
        df = df.drop('num', axis=1)

    #TO VERIFY: Instead of dropping rows that originally had NaN values in the 'ht' column, we impute the missing height values using the median height from the training dataset.
    if original_df is not None:
        # Synchronize the index before filtering
        mask = original_df.loc[df.index, 'ht'].isnull()
        df.loc[mask, 'ht'] = train_stats['ht_median']

    # Drop 'pick' column if it exists
    if 'pick' in df.columns:
        df = df.drop('pick', axis=1)

    #TO VERIFY: Instead of dropping rows with NaN values in certain columns, we use the median of the training dataset for imputation.
    columns_to_check = ['ht', 'yr', 'oreb', 'dreb',
                        'ast', 'treb', 'stl', 'pts', 'blk', 'mp',
                        'drtg', 'adrtg', 'dporpag', 'stops', 'bpm', 
                        'obpm', 'dbpm', 'gbpm', 'ogbpm', 'dgbpm']
    for column in columns_to_check:
        df[column].fillna(train_stats['medians'].get(column, df[column].median()), inplace=True)

    # Drop 'Rec_Rank' column if it exists
    if 'Rec_Rank' in df.columns:
        df = df.drop('Rec_Rank', axis=1)

    return df, train_stats

