from sklearn.decomposition import PCA

def apply_pca(data, n_components=None):
    """
    Apply PCA to the data.

    Parameters:
    - data: DataFrame, the input data.
    - n_components: int, optional (default=None), number of components to keep. 
                    If None, all components are kept.

    Returns:
    - DataFrame with principal components.
    - PCA model (so we can access attributes like explained_variance_ratio_ if needed).
    """
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    
    # Convert the principal components to a DataFrame
    column_names = [f"PC{i+1}" for i in range(principal_components.shape[1])]
    pca_df = pd.DataFrame(principal_components, columns=column_names, index=data.index)
    
    return pca_df, pca
