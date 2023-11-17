"""
Lupus: Plotting Pf2 factors and weights
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotFactors import (
    plotFactors,
    plotWeight,
)
from ..imports import import_lupus
from ..parafac2 import pf2, cwSNR
from .commonFuncs.plotUMAP import plotLabelsUMAP

import scanpy as sc
import numpy as np

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((30, 35), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    # XX = openPf2(rank, "Lupus")
    # print(XX)
    X = import_lupus()
    
    
    
    # print(X)
    
    # sc.pl.scatter(X, x='total_counts', y='pct_counts_Mitochondrial', color="SLE_status", ax=ax[0])
    # sc.pl.scatter(X, x='total_counts', y='n_genes_by_counts',color="SLE_status", ax=ax[1])
    
    # # filter for percent mito


    # X = X[X.obs['pct_counts_Mitochondrial'] < 20, :]

    # # filter for percent ribo > 0.05
    # X = X[X.obs['pct_counts_Ribosomal'] > 5, :]
    
    # print(X)
    
    # malat1 = X.var_names.str.startswith('MALAT1')
    # mito_genes = X.var_names.str.startswith('MT-')
    # hb_genes = X.var_names.str.contains('^HB[^(P)]')

    # remove = np.add(mito_genes, malat1)
    # remove = np.add(remove, hb_genes)
    # keep = np.invert(remove)

    # X = X[:,keep]

    # print(X.n_obs, X.n_vars)
    
    # # X = X[X.var != "MALAT1"]
    # genes = X.var_names != "MALAT1"
    # save = X.var_names[genes]
    # X = X[:, save]
    
    print(X)
    
    X = pf2(X, rank=40, doEmbedding=True)
    _, r2x = cwSNR(X)
    print(r2x)
    
    
    lupusStatus = X.obs[["Condition", "SLE_status"]].drop_duplicates("Condition")
    lupusStatus = lupusStatus.set_index("Condition")["SLE_status"]

    factors = [X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]]
    plotFactors(
        factors, X, ax[0:3], reorder=(0, 2), trim=(2,), cond_group_labels=lupusStatus
    )
    plotWeight(X.uns["Pf2_weights"], ax[3])

    plotLabelsUMAP(X, "Cell Type", ax[4])
    return f
