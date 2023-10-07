"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup
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

    X = ThompsonXA_SCGenes()
    X = pf2(X, "Drugs", rank=30)

    for i in range(0, 8):
        plotCmpUMAP(
            i, X.uns["Pf2_B"], X.obsm["umap"], X.obsm["projections"], ax[i]
        )

    return f
