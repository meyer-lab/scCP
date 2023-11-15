"""
Thomson: Plotting Pf2 factors and weights
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotFactors import (
    plotFactors,
    plotWeight,
)
import numpy as np
import pandas as pd
import scipy.sparse as sps
import cupy as cp
from ..imports import import_thomson, import_lupus
import seaborn as sns
from ..parafac2 import pf2, cwSNR
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 12), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    X = openPf2(rank, "Lupus")
    print(X.X)
    print(X)
    _, r2x = cwSNR(X)
    print(r2x)
    
    # # X.write('lupusa
    # """Opens weight factors and projections for one dataset for a component as numpy arrays"""
    
    # data = X.X
    # rank = 30
    # X = import_lupus()
    # data = X.X
    # data = X.X.to_memory().toarray()
    # print(np.shape(data))
    # # print(data.toarray())
    # readCount = np.zeros(np.shape(data)[0])
    # for i in range(np.shape(data)[0]):
    #     print(i)
    #     readCount[i] = np.sum((data[i, :] > 0))
    # print(readCount)
    
    # sns.histplot(data=readCount, bins=100,ax=ax[0])
        
        
    
    # for in 
    # # print(np.max(a,axis=0))
    # # print(a)
    # min = np.min(a,axis=0)
    # print(min)
    # d = X.raw.to_memory()
    # print(np.max(d,axis=0))
    # print(np.min(d,axis=0))
    
    # X = openPf2(rank, "Thomson")
    # d = X.X.to_memory()
    # print(np.max(d,axis=0))
    # print(np.min(d,axis=0))
    # d = X.raw.X.to_memory()
    # print(np.max(d,axis=0))
    # print(np.min(d,axis=0))
    # print(d)
    # print(np.shape(d))
    
    # X = cp.sparse.csr_matrix(X)
    # print(X)
    # print(cp.sparse.csr_matrix(X.X))
    # X = X.to_memory()
    
    # X = X[:, ["CD4", "MS4A1"]]
    # df = pd.DataFrame(data=X.X.data)
    # print(df)

    # X = sps.csr_array(X.X)
    # print(X.to_numpy())
    # print(X.data)
    # print(data)
 
    # print(a)
    # print(a.toarray())
    # print(X.X)
    # drugNames = groupDrugs(X.obs["Condition"])

    # factors = [X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]]
    # plotFactors(
    #     factors, X, ax[0:3], reorder=(0, 2), trim=(2,), cond_group_labels=drugNames
    # )
    # plotWeight(X.uns["Pf2_weights"], ax[3])

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

    names[names == "Everolimus (RAD001)"] = "Prostaglandin"
    names[names == "Rapamycin (Sirolimus)"] = "Prostaglandin"
    names[names == "Alprostadil"] = "mTOR Inhibitor"
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
