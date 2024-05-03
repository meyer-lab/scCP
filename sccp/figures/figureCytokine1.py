"""
Lupus: Plotting Cytokine factors and weights
"""
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotFactors import (
    plotConditionsFactors,
    plotCellState,
    plotGeneFactors,
    plotWeight,
)
from .commonFuncs.plotLupus import getSamplesObs
from ..factorization import correct_conditions
from ..imports import import_cytokine
from ..factorization import pf2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 2))

    # Add subplot labels
    subplotLabel(ax)
    """
    X = import_cytokine()
    ranks = [10, 20, 30]
    for rank in ranks:
        cytok_pf2 = pf2(X, rank)
        cytok_pf2.write_h5ad("/home/brianoj/scCP/cytok_pf2/cytok_pf2_" + str(rank))
    """
    X = read_h5ad("/home/brianoj/scCP/cytok_pf2/cytok_pf2_30")

    X.uns["Pf2_A"] = correct_conditions(X)
    stimulations = getSamplesObs(X.obs)["Condition"]

    plotConditionsFactors(X, ax[0], stimulations)
    ax[0].set(yticks=[])
    plotCellState(X, ax[1])
    plotGeneFactors(X, ax[2])
    plotWeight(X, ax[3])

    return f
