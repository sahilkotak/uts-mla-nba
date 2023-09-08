import numpy as np

def calculate_train_rolling_statistics(train_df, window=2):
    """Calculate rolling statistics and yearly growth rates for the training set."""
    # Sort values for rolling computation
    train_df = train_df.sort_values(by=['player_id', 'year'])

    # Calculate rolling statistics
    rolling_mean = train_df.groupby('player_id').apply(lambda group: group.rolling(window=window).mean()).reset_index(drop=True)
    rolling_std = train_df.groupby('player_id').apply(lambda group: group.rolling(window=window).std()).reset_index(drop=True)
    
    # Rename columns
    rolling_mean = rolling_mean.add_prefix("rolling_mean_")
    rolling_std = rolling_std.add_prefix("rolling_std_")
    
    # Calculate yearly growth rates for performance metrics
    # Using the formula: growth_rate = (current_year - previous_year) / previous_year
    yearly_growth_rates = train_df.groupby('player_id').pct_change().reset_index(drop=True)
    yearly_growth_rates = yearly_growth_rates.add_prefix("growth_rate_")
    
    # Combining the rolling statistics and growth rates with the original dataframe
    train_df = pd.concat([train_df, rolling_mean, rolling_std, yearly_growth_rates], axis=1)
    
    return train_df

def calculate_test_statistics(test_df, train_df, window=2):
    """Calculate rolling statistics and yearly growth rates for the test set based on the training set."""
    # For the test set, we'll consider the rolling statistics of the last window-1 years from the training set.
    last_years_data = train_df[train_df['year'].isin(range(2021-window+1, 2021))]

    # Calculating the rolling statistics for the training data until 2020
    train_stats = calculate_train_rolling_statistics(pd.concat([last_years_data, test_df]))

    # Filtering out the statistics for the year 2021 (test set year)
    test_stats = train_stats[train_stats['year'] == 2021].reset_index(drop=True)
    
    return test_stats
