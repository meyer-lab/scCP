"""
Lupus: Plotting weighted projections per component
"""
from anndata import read_h5ad
from .common import getSetup
from .commonFuncs.plotUMAP import plotCmpPerCellType, plotCmpUMAP


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((30, 40), (6, 5))

    X = read_h5ad("factor_cache/Lupus.h5ad", backed="r")

    cmp = [28, 21, 10, 4, 2, 1]
    for i in range(0, 30):
        plotCmpUMAP(X, i+1, ax[i], cbarMax=0.3)


    return f
