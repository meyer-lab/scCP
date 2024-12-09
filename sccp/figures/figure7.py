"""
XXX
"""

import anndata
import numpy as np
import pacmap
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
from .common import getSetup, subplotLabel
import seaborn as sns
import pandas as pd
import scipy
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import stats
import mygene
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from .common import getSetup, subplotLabel
from .commonFuncs.plotFactors import plot_gene_factors
from .commonFuncs.plotGeneral import cell_count_perc_df, rotate_xaxis, avegene_per_status
from .commonFuncs.plotPaCMAP import plot_gene_pacmap, plot_wp_pacmap


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (1, 1))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/pf2/thomson_fitted.h5ad")
    n_components = 20
    
    pc = PCA(n_components=n_components)
    pca = pc.fit(np.asarray(X.X - X.var["means"].values))
    
    geneAmount = 30
    pc_load = load_pc_loadings(pca, X)
    pf2_factors = load_pf2_loadings(X)
    
    # print(pc_load)
    # print(pf2_factors)
    
    
    df =  calculate_cross_dataframe_gene_overlap(pc_load, pf2_factors, "JS")
    # test = "Spearman"
    # df = calculate_cross_dataframe_correlation(pc_load.iloc[:, 1:], pf2_factors.iloc[:, 1:], test)
    # print(df)

    f = sns.clustermap(
        df,
        robust=True,
        vmin=-1,
        vmax=1,
        cmap='coolwarm',
        center=0,
        # row_cluster=False,
        # col_cluster=False,
           annot=True,
        figsize=(15, 15),
    )

        
    return f
    
    


def load_pc_loadings(pca, X) -> pd.DataFrame:
    """
    Load and preprocess PC loadings for a specific component.

    Args:
        pc_component (int): Principal Component number (1 or 2)
        geneAmount (int): Number of genes to consider

    Returns:
        pd.DataFrame: Processed PC loadings DataFrame
    """
    df = pd.DataFrame(data=pca.components_.T, columns=[f"PC_{i}" for i in range(1, pca.components_.T.shape[1]+1)])
    df =  df.set_index(X.var_names).reset_index().rename(columns={"index": "Gene"})
    return df

def load_pf2_loadings(X) -> pd.DataFrame:
    """
    Load and preprocess PC loadings for a specific component.

    Args:
        pc_component (int): Principal Component number (1 or 2)
        geneAmount (int): Number of genes to consider

    Returns:
        pd.DataFrame: Processed PC loadings DataFrame
    """

    df = pd.DataFrame(data=X.varm['Pf2_C'], columns=[f"Pf2_{i}" for i in range(1, X.varm['Pf2_C'].shape[1]+1)])
    df =  df.set_index(X.var_names).reset_index().rename(columns={"index": "Gene"})

    return df
    

    
    
    
def calculate_cross_dataframe_correlation(df1, df2, test):
    """
    Calculate Spearman correlation matrix between columns of two different DataFrames.
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        First input DataFrame
    df2 : pandas.DataFrame
        Second input DataFrame
    
    Returns:
    --------
    pandas.DataFrame
        Correlation matrix with columns from df1 as rows and columns from df2 as columns
    """
    # Get column names from both DataFrames
    df1_columns = list(df1.columns)
    df2_columns = list(df2.columns)
    
    # Initialize correlation matrix
    correlation_matrix = pd.DataFrame(
        index=df1_columns,
        columns=df2_columns,
        dtype=float
    )

    
    # Calculate Spearman correlations between columns of different DataFrames
    for col1 in df1_columns:
        for col2 in df2_columns:
            # Combine and remove NaN values
            valid_data = pd.concat([df1[col1], df2[col2]], axis=1).dropna()
            
            # Calculate correlation if enough data
            if len(valid_data) > 1:
                if test == "Spearman":
                    correlation, _ = scipy.stats.spearmanr(
                        valid_data.iloc[:, 0],  # Column from df1 
                        valid_data.iloc[:, 1]   # Column from df2
                    )
                elif test == "CS":
                    correlation = 1 - cosine(
                        valid_data.iloc[:, 0],  # Column from df1 
                        valid_data.iloc[:, 1]   # Column from df2
                    )
                
                correlation_matrix.loc[col1, col2] = correlation
            else:
                correlation_matrix.loc[col1, col2] = np.nan
    
    return correlation_matrix



    
def calculate_cross_dataframe_gene_overlap(df1, df2, test):
    """
    Calculate Spearman correlation matrix between columns of two different DataFrames.
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        First input DataFrame
    df2 : pandas.DataFrame
        Second input DataFrame
    
    Returns:
    --------
    pandas.DataFrame
        Correlation matrix with columns from df1 as rows and columns from df2 as columns
    """
    # Get column names from both DataFrames
    df1_columns = list(df1.columns)[1:]
    df2_columns = list(df2.columns)[1:]
    
    # Initialize correlation matrix
    correlation_matrix = pd.DataFrame(
        index=df1_columns,
        columns=df2_columns,
        dtype=float
    )

    
    # Calculate Spearman correlations between columns of different DataFrames
    for col1 in df1_columns:
        for col2 in df2_columns:
            # Combine and remove NaN values
    
            valid_data = pd.concat([df1[["Gene", col1]], df2[col2]], axis=1).dropna()
            
            # Calculate correlation if enough data
            if len(valid_data) > 1:
                if test == "JS":
                        # Top N genes
                    top_genes1 = np.argsort(np.abs(df1[col1]))
                    top_genes1 = df1["Gene"][top_genes1][-50:].to_numpy()  
                    
                    top_genes2 = np.argsort(np.abs(df2[col2]))
                    top_genes2 = df2["Gene"][top_genes2][-50:].to_numpy()  
    
                    
                    # Jaccard Similarity
                    intersection = len(set(top_genes1) & set(top_genes2))
                    union = len(set(top_genes1) | set(top_genes2))
                    jaccard_index = intersection / union
                
                correlation_matrix.loc[col1, col2] = jaccard_index
            else:
                correlation_matrix.loc[col1, col2] = np.nan
    
    return correlation_matrix
