"""
Lupus: Plot average gene expression of top/bottom genes for each component
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotGeneral import plotGenePerCategStatus
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 20), (5, 5))

    # Add subplot labels
    subplotLabel(ax)

    # Import of data
    rank = 40
    X = openPf2(rank=rank, dataName="Lupus")
    cmp = 13
    X.obs["Cell Type"] = X.obs["louvain"]
    plotGenePerCategStatus(X, cmp, rank, "Lupus", ax[0:25], geneAmount=12)



    return f
