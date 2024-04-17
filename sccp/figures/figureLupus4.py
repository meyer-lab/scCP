"""
Lupus: UMAP and boxplots of weighted projectoins per component
"""
import numpy as np
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import plotR2X


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((4, 4), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")

    ranks = [5, 10]
    
    plotR2X(X, ranks, ax[0])

   

    return f
