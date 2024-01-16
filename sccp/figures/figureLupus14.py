"""
Lupus: Plotting weighted projections per component
"""
from anndata import read_h5ad
from .common import getSetup
from .commonFuncs.plotUMAP import plotCmpPerCellType


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 8), (3, 2))

    X = read_h5ad("factor_cache/Lupus.h5ad", backed="r")

    cmp = [28, 21, 10, 4, 2, 1]
    for i, comp in enumerate(cmp):
        plotCmpPerCellType(X, comp, ax[i], outliers=False, cellType="louvain")

    return f
