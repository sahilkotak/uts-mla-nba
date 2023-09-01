# Defining a function to bin the 'team' and 'conf' columns based on basketball knowledge

def bin_team_and_conf(df):
    """
    Create two new features based on 'team' and 'conf' columns.

    Parameters:
    - df: DataFrame with the training or test data.

    Returns:
    - DataFrame with two new features 'Team_Success' and 'Conf_Level'.
    """

    # Create a copy of the dataframe to ensure no warnings and unexpected behavior
    df = df.copy()

    # Define mappings for the 'conf' column to bin them into broader categories based on quality of play
    conf_mapping = {
        'ACC': 'Power',
        'B10': 'Power',
        'B12': 'Power',
        'P12': 'Power',
        'SEC': 'Power',
        'BE': 'Power',
        'Amer': 'Mid-Major',
        'A10': 'Mid-Major',
        'WCC': 'Mid-Major',
        'MWC': 'Mid-Major',
        'CUSA': 'Mid-Major',
        'MAC': 'Minor',
        'SB': 'Minor',
        'Horz': 'Minor',
        'BW': 'Minor',
        'Ivy': 'Minor',
        'OVC': 'Minor',
        'MAAC': 'Minor',
        'Slnd': 'Minor',
        'SC': 'Minor',
        'NEC': 'Minor',
        'Pat': 'Minor',
        'BSth': 'Minor',
        'WAC': 'Minor',
        'MEAC': 'Minor',
        'SWAC': 'Minor',
        'Sum': 'Minor',
        'AE': 'Minor',
        'BSky': 'Minor',
        'ASun': 'Minor',
        'BSC': 'Minor',
    }

    # Apply the mappings to create the new 'Conf_Level' column
    df['Conf_Level'] = df['conf'].map(conf_mapping)
    
    # Calculate the average Ortg and Drtg for each team
    team_avg_ortg = df.groupby('team')['Ortg'].mean()
    team_avg_drtg = df.groupby('team')['drtg'].mean()
    
    # Define performance tiers based on Ortg and Drtg
    high_ortg_threshold = team_avg_ortg.quantile(0.75)
    low_ortg_threshold = team_avg_ortg.quantile(0.25)
    
    high_drtg_threshold = team_avg_drtg.quantile(0.75)
    low_drtg_threshold = team_avg_drtg.quantile(0.25)
    
    # Create a mapping for team performance tiers
    team_performance_tier = {}
    for team in df['team'].unique():
        avg_ortg = team_avg_ortg.get(team, 0)
        avg_drtg = team_avg_drtg.get(team, 0)
        
        if avg_ortg >= high_ortg_threshold and avg_drtg <= low_drtg_threshold:
            team_performance_tier[team] = 'High'
        elif avg_ortg <= low_ortg_threshold and avg_drtg >= high_drtg_threshold:
            team_performance_tier[team] = 'Low'
        else:
            team_performance_tier[team] = 'Medium'
            
    # Create a new feature for team performance tier
    df['team_performance_tier'] = df['team'].map(team_performance_tier)

    return df
