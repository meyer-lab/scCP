"""
Thomson: Plotting Pf2 factors and weights
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotFactors import plotFactors
from .commonFuncs.plotUMAP import plotLabelsUMAP
from ..gating import gateThomsonCells
from ..imports import import_thomson
from ..factorization import pf2
import numpy as np
import pandas as pd
from cupyx.scipy import sparse as cupy_sparse
import cupy as cp
import scipy.sparse as sps
import seaborn as sns



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((30, 30), (7, 7))

    # Add subplot labels
    subplotLabel(ax)

    rank = 30
    # X, cond = import_thomson(normalize=True)
    
    
    # sgIndex = X.obs["condition_unique_idxs"].to_numpy(dtype=int)
    # X = sps.csr_array(X.X)
    # for i in range(np.amax(sgIndex) + 1):
    #     print(cond[i])
    #     # Prepare CuPy matrix
    #     mat = cupy_sparse.csr_matrix(X[sgIndex == i], dtype=cp.float32)
    #     sum = np.sum(mat, axis=1).get().flatten()
    #     sns.histplot(data=sum, bins=30, ax=ax[i])
    #     ax[i].set(title=cond[i])
        
        
        
        
        
    # X = pf2(X, rank)
    # X = pf2(X, rank)
    X = openPf2(rank, "Thomson")
    
    drugNames = groupDrugs(X.obs["Condition"])

    factors = [X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]]
    plotFactors(
        factors, X, ax[0:3], reorder=(0, 2), trim=(2,), cond_group_labels=drugNames
    )
    gateThomsonCells(X)
    plotLabelsUMAP(X, "Cell Type", ax[3])

    return f


def groupDrugs(labels):
    """Groups drugs of similar category"""
    names = np.unique(labels)

    glucs = [
        "Triamcinolone Acetonide",
        "Loteprednol etabonate",
        "Betamethasone Valerate",
        "Budesonide",
        "Meprednisone",
    ]
    for i in glucs:
        names[names == i] = "Glucocoritcoids"

    ctrl = ["CTRL1", "CTRL2", "CTRL3", "CTRL4", "CTRL5", "CTRL6"]
    for i in ctrl:
        names[names == i] = "Control"

    names[names == "Everolimus (RAD001)"] = "mTOR Inhibitor"
    names[names == "Rapamycin (Sirolimus)"] = "mTOR Inhibitor"
    names[names == "Alprostadil"] = "Prostaglandin"
    names[names == "Cyclosporine"] = "Calcineruin Inhibitor"

    condition = [
        "Glucocoritcoids",
        "Control",
        "Prostaglandin",
        "mTOR Inhibitor",
        "Calcineruin Inhibitor",
    ]

    names = pd.Series([c if c in condition else "Other" for c in names])

    return names
