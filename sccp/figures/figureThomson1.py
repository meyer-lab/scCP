"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import plotCV, plotR2X
from .commonFuncs.plotFactors import (
    plotFactors,
    plotWeight,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import pf2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 16), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    X = ThompsonXA_SCGenes()

    X = pf2(X, "Drugs", rank=30)

    factors = [X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]]
    plotFactors(factors, X, ax[0:3], reorder=(0, 2), trim=(2,))
    plotWeight(X.uns["Pf2_weights"], ax[3])

    # plotCV(data, rank+3, trainPerc=0.75, ax=ax[2])
    # plotR2X(data, rank+3, ax=ax[3])

    return f
