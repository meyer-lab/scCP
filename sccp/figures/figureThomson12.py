"""
Thomson: XX
"""
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
)
from .commonFuncs.plotUMAP import plotLabelsUMAP
from ..gating import gateThomsonCells
from ..parafac2 import pf2
from ..imports import import_thomson
import numpy as np
from tlviz.factor_tools import factor_match_score as fms
from tensorly.cp_tensor import cp_flip_sign, CPTensor
from .commonFuncs.plotFactors import (
    plotFactors,
    plotWeight,
)



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (4, 4))

    # Add subplot labels
    subplotLabel(ax)
    rank = 30
    # X = openPf2(rank, "Thomson")
    X = import_thomson()
    X = pf2(X, "Condition", rank, random_state=1)
    Xcp = CPTensor((X.uns["Pf2_weights"], [X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]]))
    
    # XX = import_thomson()
    XX = pf2(X, "Condition", rank, random_state=3)
    XXcp = CPTensor((XX.uns["Pf2_weights"], [XX.uns["Pf2_A"], XX.uns["Pf2_B"], XX.varm["Pf2_C"]]))
    
    
    plotFactors([X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]], X, ax[0:3], reorder=(0, 2), trim=(2,))
    plotFactors([XX.uns["Pf2_A"], XX.uns["Pf2_B"], XX.varm["Pf2_C"]], XX, ax[3:6], reorder=(0, 2), trim=(2,))
    

    

    
    print(fms(Xcp, XXcp, consider_weights=True, skip_mode=None))
    print(fms(Xcp, XXcp, consider_weights=True, skip_mode=0))
    print(fms(Xcp, XXcp, consider_weights=False, skip_mode=None))
    print(fms(Xcp, XXcp, consider_weights=False, skip_mode=0))
    
    
    
    
       
    print(fms(Xcp, Xcp, consider_weights=True, skip_mode=None))
    print(fms(Xcp, Xcp, consider_weights=True, skip_mode=0))
    print(fms(Xcp, Xcp, consider_weights=False, skip_mode=None))
    print(fms(Xcp, Xcp, consider_weights=False, skip_mode=0))
    
    
    # for i in range(X.uns["Pf2_A"].shape[1]):
    #     A = np.transpose(X.uns["Pf2_A"][:, i]) @ XX.uns["Pf2_A"][:, i]
    #     B = np.transpose(X.obsm["weighted_projections"][:, i]) @ XX.obsm["weighted_projections"][:, i]
    #     C = np.transpose(X.varm["Pf2_C"][:, i]) @ XX.varm["Pf2_C"][:, i]
        # total.append(np.abs(A * B * C))
    #     total[i] = np.abs(A * B * C)
        
    # print(len(total))   
    # fms = np.sum(total)/X.uns["Pf2_A"].shape[1]
    # print(fms)
    return f
    
        
        