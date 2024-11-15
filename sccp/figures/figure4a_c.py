"""
Figure 4a_c
"""

from anndata import read_h5ad

from ..factorization import correct_conditions
from .common import getSetup, subplotLabel
from .commonFuncs.plotLupus import plot_roc_fourthbatch
from .commonFuncs.plotPaCMAP import plot_labels_pacmap
import numpy as np
from scipy import sparse
import pandas as pd
import scanpy as sc

# from .commonFuncs.plotLupus import plot_accuracy_ranks_lupus
# from .commonFuncs.plotGeneral import plot_r2x


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((9, 4), (1, 3))
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    print(X)
    XX = aggregate_anndata(X, celltype_col='Cell Type', method='average')
    print(XX)
    
    # Can you create a function that gets the cell type compositions per cell type across across all conditions 
    
    

    # ranks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # plot_accuracy_ranks_lupus(X, ranks, ax[0], error_metric="roc_auc")
    # ax[0].set(xticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    # x = [0, 50]
    # y = [0.84, 0.84]
    # ax[0].plot(x, y, linestyle="--")

    # X.uns["Pf2_A"] = correct_conditions(X)
    # plot_roc_fourthbatch(X, ax[1])

    # plot_labels_pacmap(X, "Cell Type2", ax[2])

    return f



def aggregate_anndata(adata, 
                              celltype_col='cell_type',
                              condition_col='Condition',
                              method='average',
                              layer=None,
                              min_cells=5):
    '''
    Aggregates gene expression by both cell types and conditions.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing gene expression data and metadata
    celltype_col : str, default='cell_type'
        Column name in adata.obs containing cell type annotations
    condition_col : str, default='condition'
        Column name in adata.obs containing condition annotations
    method : str, default='average'
        Method to aggregate expression:
        - 'average': Mean expression per group
        - 'nn_cell_fraction': Fraction of cells with non-zero expression
    layer : str, optional
        Layer in AnnData to use for aggregation. If None, uses .X
    min_cells : int, default=5
        Minimum number of cells required for a cell type-condition group
        
    Returns
    -------
    AnnData
        New AnnData object with aggregated data where:
        - Observations are cell type-condition combinations
        - Variables are genes
        - .obs contains cell type and condition annotations
    '''
   
    X = adata.X
    
    # Convert to dense if sparse
    if sparse.issparse(X):
        X = X.toarray()
    
    # Create combination groups
    # adata.obs['group'] = adata.obs[celltype_col].astype(str) + "_" + adata.obs[condition_col].astype
    
    # Get unique combinations
    cell_types = adata.obs[celltype_col].unique()
    print(cell_types)
    print(len(cell_types))
    conditions = adata.obs[condition_col].unique()
    
    print(len(conditions))
    
    # Initialize storage for aggregated data
    aggregated_data = []
    valid_groups = []
    cell_counts = []
    cell_type_list = []
    condition_list = []
    sle_status_list = []
    
    # Aggregate for each combination
    for ct in cell_types:
        for cond in conditions:
            # Get mask for current group
            mask = (adata.obs[celltype_col] == ct) & (adata.obs[condition_col] == cond)
            group_data = X[mask]
            
            # Check if enough cells
            if group_data.shape[0] >= min_cells:
                if method == 'average':
                    agg_values = np.mean(group_data, axis=0)
                elif method == 'nn_cell_fraction':
                    agg_values = np.mean(group_data > 0, axis=0)
                
                aggregated_data.append(agg_values)
                valid_groups.append(f"{ct}_{cond}")
                cell_counts.append(group_data.shape[0])
                cell_type_list.append(ct)
                condition_list.append(cond)
                sle_status_list.append(group_data.obs['SLE Status'].unique()[0])
    
    # Convert to array
    aggregated_data = np.array(aggregated_data)
    
    # Create observation DataFrame
    obs_df = pd.DataFrame({
        'cell_type': cell_type_list,
        'condition': condition_list,
        'n_cells': cell_counts,
        'sle_status': sle_status_list
    }, index=valid_groups)
    
    # Create new AnnData object
    aggregated_adata = sc.AnnData(
        X=aggregated_data,
        obs=obs_df,
        var=adata.var.copy()
    )
    
    # Add metadata
    aggregated_adata.uns['aggregation_method'] = method
    aggregated_adata.uns['min_cells'] = min_cells
    if layer:
        aggregated_adata.uns['aggregation_layer'] = layer
    
    return aggregated_adata


def get_cell_type_compositions(adata, celltype_col, condition_col):
    """Get cell type compositions per cell type across all conditions."""
    compositions = {}
    for condition in adata.obs[condition_col].unique():
        condition_data = adata[adata.obs[condition_col] == condition]
        compositions[condition] = condition_data.obs[celltype_col].value_counts(normalize=True)
    return compositions