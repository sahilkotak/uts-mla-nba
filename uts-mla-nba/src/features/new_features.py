# Function to generate new features based on the given definitions
def generate_new_features(df):
    """
    Generate new features based on basketball-specific calculations.

    Parameters:
    - df: DataFrame containing the raw basketball statistics.

    Returns:
    - DataFrame with the new features added.
    """

    # Net Rating
    df['Net_Rating'] = df['Ortg'] - df['drtg']

    # Scoring Efficiency
    df['Scoring_Efficiency'] = df['pts'] / (df['FGA'] + 0.44 * df['FTA'])

    # Rebound Rate
    df['Rebound_Rate'] = df['TRB'] / df['GP']

    # Assist to Turnover Ratio
    df['Ast_To_Turnover_Ratio'] = df['AST'] / df['TO_per']

    # Steal to Turnover Ratio
    df['Stl_To_Turnover_Ratio'] = df['STL'] / df['TO_per']

    # Points Per Minute
    df['Points_Per_Minute'] = df['PTS'] / df['MP']

    # Impact Score
    df['Impact_Score'] = df['PTS'] + df['AST'] + df['REB'] - (df['FTA'] - df['FTM'])

    # Team Average ORtg and DRtg (assuming we have a 'Team' column to group by)
    if 'Team' in df.columns:
        df['Team_Avg_ORtg'] = df.groupby('Team')['ORtg'].transform('mean')
        df['Team_Avg_DRtg'] = df.groupby('Team')['DRtg'].transform('mean')

    # Versatility Index (simplified to be a combination of PTS, AST, and REB)
    df['Versatility_Index'] = df['PTS'] + df['AST'] + df['REB']

    return df