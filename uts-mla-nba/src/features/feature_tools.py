def generate_features(data):
    """
    Generate new features using Featuretools.
    
    Parameters:
    - data: Preprocessed DataFrame with training data.
    
    Returns:
    - DataFrame with original and new features.
    """
    import featuretools as ft

    # Make an entityset and add the entity (table)
    es = ft.EntitySet(id="players")
    
    # Specify which column is the unique identifier in the data
    data.ww.init(index="player_id")
    
    # Add the dataframe to the entityset
    es.add_dataframe(dataframe_name="data", dataframe=data)

    # Run deep feature synthesis with transformation primitives
    features, feature_names = ft.dfs(entityset=es, target_dataframe_name="data",
                                     trans_primitives=["add_numeric", "multiply_numeric"],
                                     verbose=True, max_depth=2)

    return features
