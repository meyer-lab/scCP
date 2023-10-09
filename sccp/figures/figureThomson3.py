"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
import numpy as np
from .common import subplotLabel, getSetup, openPf2, openPf2
from .commonFuncs.plotUMAP import (
    plotCmpUMAP,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import pf2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    # UMAP dimension reduction
    _, _, projs = openPf2(rank, "Thomson")

    for i in range(0, 8):
        plotCmpUMAP(
            i, X.uns["Pf2_B"], X.obsm["umap"], X.obsm["projections"], ax[i]
        )

    return f
