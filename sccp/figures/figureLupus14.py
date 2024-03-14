"""
Lupus: Plotting weighted projections per component
"""
from anndata import read_h5ad
from .common import getSetup
from .commonFuncs.plotUMAP import plotCmpPerCellTypeStatus
from .commonFuncs.plotUMAP import plotCmpPerCellTypeStatus, plotCmpUMAP


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((14, 28), (8, 5))

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")

    comps = [3, 27]
    for i, cmp in enumerate(comps):
        plotCmpPerCellTypeStatus(X, cmp, ax[(2 * i)], cellType="Cell Type2")
        plotCmpUMAP(X, cmp, ax[(2 * i) + 1], cbarMax=0.3)
        

    return f
