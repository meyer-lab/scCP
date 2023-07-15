"""
S3: Examining cell state for Pf2 on lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: visualize the cell state compostition by cell type/UMAP

import numpy as np
from .common import (
    subplotLabel,
    getSetup,
    plotCmpUMAP
)
from ..imports.scRNA import tensorFy
from ..parafac2 import parafac2_nd
import pandas as pd
import numpy as np
import tensorly as tl
import seaborn as sns
import anndata
import umap
from sklearn.decomposition import PCA
from matplotlib import pyplot



####################################################################################
## LOAD IN DATA -- CAN BE MOVED TO imports/scRNA 
####################################################################################

def load_lupus_data(third_axis= "ind_cov", n_rand = 100):
    """Import Lupus PBMC dataset."""
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus.h5ad")
    
    # reorder X so that all of the patients are in alphanumeric order. this is important
    # so that we can steal cell typings at this point
    obsV = X.obs_vector('ind_cov')
    sgUnique, sgIndex = np.unique(obsV, return_inverse=True)

    ann_data_objects = [X[sgIndex == sgi, :] for sgi in range(len(sgUnique))]

    X = anndata.concat(ann_data_objects, axis = 0)

    # select n_rand random patients; evenly split between lupus and SLE
    np.random.seed(42)
    SLE_patients = X.obs["ind_cov"][X.obs["SLE_status"] == "SLE"].to_numpy()
    healthy_patients = X.obs["ind_cov"][X.obs["SLE_status"] == "Healthy"].to_numpy()
    random_patients = np.random.choice(SLE_patients, int(n_rand/2), replace=False).tolist() + np.random.choice(healthy_patients, int(n_rand/2), replace=False).tolist()

    X = X[X.obs['ind_cov'].isin(random_patients), :]

    # get cell types -> only stay true if order of cells doesn't change after this
    cell_types = X.obs['cg_cov'].reset_index(drop=True)

    # get color mapping for patients by SLE status

    status = X.obs[["ind_cov","SLE_status"]].sort_values(by= "ind_cov").drop_duplicates("ind_cov")

    lut = {'SLE': 'c', 'Healthy': 'm'}
    row_colors = status['SLE_status'].map(lut)

    assert np.all(np.isfinite(X.X.data)) # this should be true

    return tensorFy(X, third_axis), cell_types, row_colors

####################################################################################



def plotUMAP_ct(labels, pf2Points, projs, ax):
    """Scatterplot of UMAP visualization labeled by cell type"""
    allP = np.concatenate(projs, axis=0)
    plot = umap.plot.points(pf2Points, 
                            labels = labels, 
                            theme='viridis', 
                            ax=ax)
    ax.set(
        ylabel="UMAP2",
        xlabel="UMAP1",
        title="Pf2-Based Decomposition: Label Cell Types")
 

def plotCellStateViolins(projections, cell_types, cell_state: int, ax):
    all_cell_projs = pd.DataFrame(tl.concatenate(list(projections), axis=0))
    cell_state_n = pd.concat([all_cell_projs.iloc[:, (cell_state - 1)], cell_types], axis = 1)
    cell_state_n.columns.values[0] = "contribution"

    sns.violinplot(data = cell_state_n,
                   x = "cg_cov",
                   y = 'contribution',
                   hue = 'cg_cov',
                   dodge = False,
                   ax = ax)
    
    ax.set_title('Cell Type Contrib to Cell State ' + str(cell_state))
    ax.tick_params(axis="x", rotation=90)
    ax.get_legend().remove()

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of data
    data, cell_types, _ = load_lupus_data() # don't need to get patient color mappings
    rank = 30
    cellState = 28; cmp = 28

    # run pf2
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
        verbose=True,
    )

    # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))

    # PCA dimension reduction
    pc = PCA(n_components=rank)
    pcaPoints = pc.fit_transform(data.unfold())
    pcaPoints = umap.UMAP(random_state=1).fit(pcaPoints)


    plotUMAP_ct(cell_types, pf2Points, projs, ax[0])
    plotCmpUMAP(cellState, cmp, factors, pf2Points, projs, ax[1])
    plotCellStateViolins(projs, cell_types, cellState, ax[2])


    return f