"""
Figure 5f_j
"""

import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from tensorly.decomposition._parafac2 import (
    _parafac2_reconstruction_error, _project_tensor_slices)

from .common import getSetup, subplotLabel
from .commonFuncs.plotPaCMAP import plot_wp_pacmap
import tlviz
from parafac2.parafac2 import parafac2_nd, store_pf2
from ..factorization import pf2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/pf2/thomson_fitted.h5ad")
    print(X)
    rank_vec = np.arange(2, 40, 2)
    plotCC(X, rank_vec, ax[0])


    return f


def plotCC(X, rank_vec, ax):
    """Creates R2X plot for parafac2 tensor decomposition"""
    
    pf2_error = np.empty(len(rank_vec))
    core_consist = pf2_error.copy()
    
    tensor = anndata_to_tensor(X, "condition_unique_idxs")

    # Collect Pf2 results
    for i, rank in enumerate(rank_vec):
        X = pf2(
                X,
                rank=rank,
                doEmbedding=False,
            )

        # Core consistency 
        print(tensor.shape)
        print(X.obsm["projections"].shape)
        projs = proj_tensor(X, X.obsm["projections"])
        print(projs.shape)
        projected_tensor = _project_tensor_slices(tensor, projs)
        print(projected_tensor.shape)
        projected_tensor_nD = np.reshape(
            projected_tensor, (*tensor.shape[0:-2], rank, tensor.shape[-1])
        )
        print(projected_tensor_nD.shape)
        factors = [X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]]
        
        core_consist[i] = tlviz.model_evaluation.core_consistency((X.uns["Pf2_weights"], factors), projected_tensor_nD, normalised=True)
        
    df = pd.DataFrame(data=core_consist/100, columns=["Core Consistency"])
    df["Rank"] = rank_vec
    
    sns.lineplot(data=df, x="Rank", y="Core Consistency", ax=ax)
        
    
    
def anndata_to_tensor(X_in: anndata.AnnData, type: str,) -> np.ndarray:
    # Index dataset to a list of conditions
    sgIndex = X_in.obs["condition_unique_idxs"].to_numpy(dtype=int)
    
    # Determine the maximum number of samples per condition
    max_samples = np.amax(np.bincount(sgIndex))
    
    # Determine the feature size (2nd dimension)
    feature_size = X_in.shape[1]

    # Initialize a 3D numpy array filled with zeros
    padded_array = np.zeros((np.amax(sgIndex) + 1, max_samples, feature_size), dtype=np.float32)

    for i in range(np.amax(sgIndex) + 1):
        # Extract data for the current condition
        if isinstance(X_in.X, np.ndarray):
            condition_data = X_in.X[sgIndex == i]
        else:
            condition_data = X_in.X[sgIndex == i].toarray()

        # Copy the data into the padded array
        num_samples = condition_data.shape[0]
        padded_array[i, :num_samples, :] = condition_data

    return padded_array


def proj_tensor(X_in: anndata.AnnData, projection_matrix: np.ndarray) -> np.ndarray:
    """
    Reorganizes the projection matrix into a 3D tensor grouped by unique conditions.

    Parameters:
        X_in (anndata.AnnData): Input AnnData object containing observation metadata.
        projection_matrix (np.ndarray): Projection matrix to reorder.

    Returns:
        np.ndarray: 3D numpy array of shape (num_conditions, max_samples, projection_size).
    """
    # Index dataset to a list of conditions
    sgIndex = X_in.obs["condition_unique_idxs"].to_numpy(dtype=int)
    
    # Determine the maximum number of samples per condition
    max_samples = np.max(np.bincount(sgIndex))
    
    # Determine the projection size (2nd dimension of the projection matrix)
    projection_size = projection_matrix.shape[1]

    # Initialize a 3D numpy array filled with zeros
    padded_array = np.zeros((np.max(sgIndex) + 1, max_samples, projection_size), dtype=np.float32)

    for i in range(np.max(sgIndex) + 1):
        # Find the indices of the current condition
        condition_indices = np.where(sgIndex == i)[0]

        # Reorder the projection matrix rows corresponding to the current condition
        condition_data = projection_matrix[condition_indices, :]

        # Copy the reordered data into the padded array
        num_samples = condition_data.shape[0]
        padded_array[i, :num_samples, :] = condition_data

    return padded_array
