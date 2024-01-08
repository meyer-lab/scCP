"""
Lupus: UMAP and boxplots of weighted projectoins per component
"""
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotUMAP import plotCmpPerCellType, plotCmpUMAP
import numpy as np
from ..imports import import_lupus


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    #ax, f = getSetup((15, 15), (3, 3))
    ax, f = getSetup((30, 30), (8, 8))

    # Add subplot labels
    #subplotLabel(ax)
    import_lupus()

    X = read_h5ad("factor_cache/Lupus.h5ad", backed="r")

    comps = [3, 15, 21, 28]
    comps = np.arange(0, 30) + 1
    for i, cmp in enumerate(comps):
        plotCmpPerCellType(X, cmp, ax[(2 * i)])
        plotCmpUMAP(X, cmp, ax[(2 * i) + 1], cbarMax=0.3)
    return f
