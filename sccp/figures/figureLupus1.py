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
import numpy as np



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 2))
    # Add subplot labels
    subplotLabel(ax)
    rank = 40
    X = openPf2(rank, "Lupus")
    # X = import_lupus() 
    data = X.to_df()
    
    print(np.max(data,axis=0).values)
    print(np.min(data,axis=0).values)
    
    
    # for cohort in np.unique(X.obs["Processing_Cohort"]):
    #     XX = X[(X.obs["Processing_Cohort"] == cohort)]
    #     print(sps.csr_array(X.X))
        # XX = pf2(XX, rank=rank)
        # _, r2x = cwSNR(XX)
        # print(cohort)
        # print(r2x)
    
    # # X.write('lupusallgenes.h5ad', compression="gzip")
    # print(X)
    # lupusStatus = X.obs[["Condition", "SLE_status"]].drop_duplicates("Condition")
    # lupusStatus = lupusStatus.set_index("Condition")["SLE_status"]
    # factors = [X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]]
    # plotFactors(
    #     factors, X, ax[0:3], reorder=(0, 2), trim=(2,), cond_group_labels=lupusStatus
    # )
    # plotWeight(X.uns["Pf2_weights"], ax[3])
    return f