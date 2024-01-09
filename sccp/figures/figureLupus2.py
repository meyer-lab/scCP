"""
Lupus: UMAP labeled by cell type
"""
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotUMAP import plotLabelsUMAP


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (2, 1))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("factor_cache/Lupus.h5ad", backed="r")

    plotLabelsUMAP(X, "louvain", ax[0])
    plotLabelsUMAP(X, "SLE_status", ax[1])

    return f
