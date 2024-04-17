"""
Thomson: Plotting weighted projections per component
"""
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotUMAP import (
    plotCmpUMAP,
)

from .commonFuncs.plotGeneral import (
    plotR2X,
)
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((3, 3), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad", backed="r")

    ranks = [5, 10, 15, 20, 25, 30]
    
    
    plotR2X(X, ranks, ax[0])
    

    return f
