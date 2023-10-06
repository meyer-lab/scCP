"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import plotCV, plotR2X
from .commonFuncs.plotFactors import (
    plotFactors,
    plotWeight,
)
from ..imports.scRNA import ThompsonXA_SCGenes, tensorFy
from parafac2 import parafac2_nd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 16), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    data = tensorFy(data, "Drugs")

    weight, factors, _, _ = parafac2_nd(
        data,
        rank=30,
        random_state=1,
    )

    plotFactors(factors, data, ax[0:3], reorder=(0, 2), trim=(2,))
    plotWeight(weight, ax[3])

    # plotCV(data, rank+3, trainPerc=0.75, ax=ax[2])
    # plotR2X(data, rank+3, ax=ax[3])

    return f
