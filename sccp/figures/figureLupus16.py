"""
Thomson: Creative ways to visualize single cells 
"""
import numpy as np
from .common import getSetup, openPf2
from .commonFuncs.plotUMAP import plotCmpPerCellType, plotLabelsUMAP2, plotLabelsUMAP
from ..gating import gateThomsonCells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 8), (1, 2))

    rank = 40
    X = openPf2(rank, "Lupus")

    # gateThomsonCells(X)

    # glucs = [
    #     "Betamethasone Valerate",
    #     "Loteprednol etabonate",
    #     "Budesonide",
    #     "Triamcinolone Acetonide",
    #     "Meprednisone",
    # ]
    # XX = X.copy(
    cmp1 = 13
    cmp2 = 26
    # plotLabelsUMAP(X, labelType="SLE_status",  ax=ax[1], cmp1=cmp1, cmp2=cmp2)
    plotLabelsUMAP2(X, labelType="SLE_status",  ax=ax[1], cmp1=cmp1, cmp2=cmp2)
    # ax[1].set(xlabel=f"Weight Proj Cmp. {cmp1}", ylabel=f"Weight Proj Cmp. {cmp2}")
    ax[1].set(xlabel=f"Gene Cmp. {cmp1}", ylabel=f"Gene Cmp. {cmp2}")


    # plotCellCmpUMAP(X, cmp1=1, cmp2=30, labelType="Cell Type", ax=ax[0])
     # plotCellCmpUMAP(X, cmp1=1, cmp2=30, labelType="Condition", ax=ax[1], condition=drugs)
    # plotCellCmpUMAP(X, cmp1=5, cmp2=6, labelType="Cell Type", ax=ax[2])

    return f
    
    