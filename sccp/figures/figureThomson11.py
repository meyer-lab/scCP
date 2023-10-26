"""
Thomson: Creative ways to visualize single cells 
"""
import numpy as np
from .common import getSetup, openPf2
from .commonFuncs.plotUMAP import plotCmpPerCellType, plotLabelsUMAP
from ..gating import gateThomsonCells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 3))

    rank = 30
    X = openPf2(rank, "Thomson")

    gateThomsonCells(X)
    
    glucs = [
        "Betamethasone Valerate",
        "Loteprednol etabonate",
        "Budesonide",
        "Triamcinolone Acetonide",
        "Meprednisone",
    ]
    XX = X.copy()
    
    plotLabelsUMAP(XX, labelType="Condition",  ax=ax[1], condition=glucs, conditionName="Gluco", cmp1=1, cmp2=30)
    
    
    
    

    # # a
    
    # plotCellCmpUMAP(X, cmp1=1, cmp2=30, labelType="Cell Type", ax=ax[0])
    # plotCellCmpUMAP(X, cmp1=1, cmp2=30, labelType="Condition", ax=ax[1], condition=drugs)
    # plotCellCmpUMAP(X, cmp1=5, cmp2=6, labelType="Cell Type", ax=ax[2])

    return f




