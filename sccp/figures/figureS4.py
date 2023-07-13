"""
S4: Sean's bold attempt at Pf2 on the lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup,
    flattenData,
    plotDrugUMAP,
    plotGeneUMAP,
    plotCmpUMAP,
)
from ..parafac2 import parafac2_nd, Pf2X
import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
import seaborn as sns
from os.path import dirname
from scipy.stats import linregress
import anndata
import scipy
from .common import subplotLabel, getSetup, plotFactors
import warnings
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import tensorly as tl




# load data (need to change filepath once dataset gets uploaded to opt/...)
# lupus_data = anndata.read_h5ad("/home/seanp/scCP/GSE174188_CLUES1_adjusted.h5ad")

# get rid of warnings <3 (this lowkey doesn't work)
warnings.filterwarnings("once")


####################################################################################
## LOAD IN DATA -- CAN BE MOVED TO imports/scRNA.py
####################################################################################

# from imports/scRNA.py
def tensorFy(annD: anndata.AnnData, obsName: str) -> Pf2X:
    obsV = annD.obs_vector(obsName)
    sgUnique, sgIndex = np.unique(obsV, return_inverse=True)

    data = [annD[sgIndex == sgi, :].X.toarray() for sgi in range(len(sgUnique))]

    return Pf2X(data, sgUnique, annD.var_names)


def lupus_data(third_axis= "ind_cov", n_rand = 30):
    """Import Thompson lab PBMC dataset."""
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus.h5ad")

    # select n_rand random patients; evenly split between lupus and SLE
    np.random.seed(42)
    SLE_patients = X.obs["ind_cov"][X.obs["SLE_status"] == "SLE"].to_numpy()
    healthy_patients = X.obs["ind_cov"][X.obs["SLE_status"] == "Healthy"].to_numpy()
    random_patients = np.random.choice(SLE_patients, int(n_rand/2), replace=False).tolist() + np.random.choice(healthy_patients, int(n_rand/2), replace=False).tolist()

    X = X[X.obs['ind_cov'].isin(random_patients), :]

    # get cell types
    cell_types = X.obs['cg_cov'].reset_index(drop=True)

    assert np.all(np.isfinite(X.X.data)) # yeah this should be true

    # X.X /= np.sum(X.X, axis=0) don't need to normalize because "we" already did <3

    # Center the genes [already done]
    # X.X -= np.mean(X.X, axis=0)

    # Assign cells a count per-experiment so we can reindex
    return tensorFy(X, third_axis), cell_types

####################################################################################

rank = 30

# hello hello hello

lupus_tensor, cell_types = lupus_data()

_, factors, projs, _ = parafac2_nd(lupus_tensor, 
                                   rank = rank, 
                                   n_iter_max= 20,
                                   random_state = 1, 
                                   verbose=True)


# dataDF, projDF, _ = flattenData(lupus_tensor, factors, projs)

# UMAP dimension reduction
# pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))

# PCA dimension reduction
# pc = PCA(n_components=rank)
# pcaPoints = pc.fit_transform(lupus_tensor.unfold())
# pcaPoints = umap.UMAP(random_state=1).fit(pcaPoints)

# genes = ["CCR7"]

def reorder_table(projs):
    """Reorder a table's rows using heirarchical clustering"""
    assert projs.ndim == 2
    Z = sch.linkage(projs, method="centroid", optimal_ordering=True)
    index = sch.leaves_list(Z)
    return projs[index, :], index

def plotLupusFactors(factors, data: Pf2X, og_data, axs, reorder=tuple(), trim=tuple(), saveGenes=False):
    """Plots parafac2 factors."""
    pd.set_option('display.max_rows', None)
    rank = factors[0].shape[1]
    xticks = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    for i in range(3):
        # The single cell mode has a square factors matrix
        if i == 0:
            yt = data.condition_labels
            print(yt, type(yt))
            # set rowcolors as SLE status
            status = og_data.obs[["ind_cov","SLE_status"]][og_data.obs['ind_cov'].isin(data.condition_labels)].sort_values(by= "ind_cov").drop_duplicates("ind_cov")

            lut = {'SLE': 'c', 'Healthy': 'm'}
            row_colors = status['SLE_status'].map(lut)

            title = "Components by Patient"
        elif i == 1:
            yt = [f"Cell State {i}" for i in np.arange(1, rank + 1)]
            title = "Components by Cell State"
        else:
            yt = data.variable_labels
            title = "Components by Gene"

        X = factors[i]

        if i in trim:
            max_weight = np.max(np.abs(X), axis=1)
            kept_idxs = max_weight > 0.08
            X = X[kept_idxs]
            yt = yt[kept_idxs]

        if i in reorder:
            X, ind = reorder_table(X)
            yt = yt[ind]
            if i == 0:
                row_colors = row_colors[ind]

        sns.heatmap(
                data=X,
                xticklabels=xticks,
                yticklabels=yt,
                ax=axs[i],
                center=0,
                cmap=cmap,
            )

        if i == 0:
            # add little boxes
            axs[i].tick_params(axis='y', which='major', pad=20, length=0) # extra padding to leave room for the row colors
            for iii, color in enumerate(row_colors):
                axs[i].add_patch(plt.Rectangle(xy=(-0.05, iii), width=0.05, height=1, color=color, lw=0,
                             transform=axs[i].get_yaxis_transform(), clip_on=False))


        axs[i].set_title(title)
        axs[i].tick_params(axis="y", rotation=0)

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), # fig size
                     (2, 2) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    all_cell_projs = pd.DataFrame(tl.concatenate(list(projs), axis=0))
    cell_state_28 = pd.concat([all_cell_projs.iloc[:, 27], cell_types], axis = 1)
    cell_state_28.columns.values[0] = "contribution"
    print(cell_state_28)

    #plotFactors(factors, lupus_tensor, ax[0:3], reorder=(0, 2), trim=(2,), saveGenes=False)
    plotLupusFactors(factors, lupus_tensor, anndata.read_h5ad("/opt/andrew/lupus/lupus.h5ad"), ax[0:3], reorder = (0,2), trim=(2,))

    # examine the 28th cell state
    sns.boxplot(data = cell_state_28,
                   x = "cg_cov",
                   y = 'contribution',
                   hue = 'cg_cov',
                   width= 4,
                   ax = ax[3])
    
    ax[3].set_title('Cell Type Contrib to Cell State 28')
    ax[3].tick_params(axis="x", rotation=90)
    ax[3].get_legend().remove()

    # print("DataDF: \n\n", dataDF)
    # print("ProjDF: \n\n", projDF)
    
    # plotGeneUMAP(genes, "Pf2", pf2Points, dataDF, ax[0:])
    # plotGeneUMAP(genes, "PCA", pcaPoints, dataDF, ax[2:])

    return f
