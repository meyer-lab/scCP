"""
Lupus: UMAP and boxplots of weighted projectoins per component
"""
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotUMAP import plotCmpPerCellType, plotCmpUMAP


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 8), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("factor_cache/Lupus.h5ad", backed="r")

    comps = [28, 21, 10, 4, 2, 1]
    for i, cmp in enumerate(comps):
        plotCmpUMAP(X, cmp, ax[i], cbarMax=0.3)
    return f
