"""
S2: Initial Attempt at Pf2 on the lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: test Pf2 on lupus data, get visualizations for factor matrices

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup,
    reorder_table
)
from ..parafac2 import parafac2_nd, Pf2X
from ..imports.scRNA import tensorFy
import pandas as pd
import numpy as np
import seaborn as sns
import anndata
from .common import subplotLabel, getSetup
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import tensorly as tl


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
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # REMOVE THIS SECTION TO HAVE ALL INDIVIDUALS INCLUDED (and n_rand arg in function def)
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    np.random.seed(42)
    SLE_patients = X.obs["ind_cov"][X.obs["SLE_status"] == "SLE"].to_numpy()
    healthy_patients = X.obs["ind_cov"][X.obs["SLE_status"] == "Healthy"].to_numpy()
    random_patients = np.random.choice(SLE_patients, int(n_rand/2), replace=False).tolist() + np.random.choice(healthy_patients, int(n_rand/2), replace=False).tolist()

    X = X[X.obs['ind_cov'].isin(random_patients), :]
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # get cell types -> only stay true if order of cells doesn't change after this
    cell_types = X.obs['cg_cov'].reset_index(drop=True)

    # get color mapping for patients by SLE status

    status = X.obs[["ind_cov","SLE_status"]].sort_values(by= "ind_cov").drop_duplicates("ind_cov")

    lut = {'SLE': 'c', 'Healthy': 'm'}
    row_colors = status['SLE_status'].map(lut)

    assert np.all(np.isfinite(X.X.data)) # this should be true

    return tensorFy(X, third_axis), cell_types, row_colors

####################################################################################

def plotLupusFactors(factors, data: Pf2X, row_colors, axs, reorder=tuple(), trim=tuple()):
    """Plots parafac2 factors for lupus dataset."""
    pd.set_option('display.max_rows', None)
    rank = factors[0].shape[1]
    xticks = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    for i in range(3):
        # The single cell mode has a square factors matrix
        if i == 0:
            yt = data.condition_labels

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
            # add little boxes to denote SLE/healthy rows
            axs[i].tick_params(axis='y', which='major', pad=20, length=0) # extra padding to leave room for the row colors
            for iii, color in enumerate(row_colors):
                axs[i].add_patch(plt.Rectangle(xy=(-0.05, iii), width=0.05, height=1, color=color, lw=0,
                             transform=axs[i].get_yaxis_transform(), clip_on=False))


        axs[i].set_title(title)
        axs[i].tick_params(axis="y", rotation=0)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), # fig size
                     (2, 2) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    rank = 30

    lupus_tensor, _, row_colors = load_lupus_data() # don't need to grab cell types here

    _, factors, _, _ = parafac2_nd(lupus_tensor, 
                                    rank = rank, 
                                    n_iter_max= 20,
                                    random_state = 1, 
                                    verbose=True)

    plotLupusFactors(factors, lupus_tensor, row_colors, ax[0:3], reorder = (0,2), trim=(2,))


    return f