"""
Plots all top and bottom genes for Thomson dataset
"""
from .common import getSetup, openPf2
from .commonFuncs.plotUMAP import plotCmpPerCellType
import numpy as np
from ..imports.gating import gateThomsonCells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((14, 20), (5, 6))

    rank = 30
    X = openPf2(rank, "Thomson")

    # TODO: Fix
    # weightedProjDF["Cell Type"] = gateThomsonCells()

    component = np.arange(1, rank + 1, 1)
    for i, comp in enumerate(component):
        plotCmpPerCellType(weightedProjDF, comp, ax[i], outliers=False)

    return f
