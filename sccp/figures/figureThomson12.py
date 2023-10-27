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


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (4, 4))

    # Add subplot labels
    subplotLabel(ax)
    rank = 30
    X = openPf2(rank, "Thomson")
    factors = [X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]]
    
    XX = import_thomson()
    # XX.X = XX.X
    XX = pf2(XX, "Condition", rank, random_state=3)
    
    # total = []
    
    total = np.empty(X.uns["Pf2_A"].shape[1], dtype=float)
    
    
    for i in range(X.uns["Pf2_A"].shape[1]):
        A = np.transpose(X.uns["Pf2_A"][:, i]) @ XX.uns["Pf2_A"][:, i]
        B = np.transpose(X.obsm["weighted_projections"][:, i]) @ XX.obsm["weighted_projections"][:, i]
        C = np.transpose(X.varm["Pf2_C"][:, i]) @ XX.varm["Pf2_C"][:, i]
        # total.append(np.abs(A * B * C))
        total[i] = np.abs(A * B * C)
        
    print(len(total))   
    fms = np.sum(total)/X.uns["Pf2_A"].shape[1]
    print(fms)
    return f
    
        
        