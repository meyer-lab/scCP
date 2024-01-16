"""
Lupus: Plots all top and bottom genes Pf2 weights
"""
from anndata import read_h5ad
from .common import getSetup
from .commonFuncs.plotGeneral import plotGeneFactors
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 5), (3, 4))

    X = read_h5ad("factor_cache/Lupus.h5ad", backed="r")

    cmp = [28, 21, 10, 4, 2, 1]
    for i, comp in enumerate(cmp):
        plotGeneFactors(comp, X, ax[2 * i], geneAmount=10, top=True)
        plotGeneFactors(comp, X, ax[2 * i + 1], geneAmount=10, top=False)

    return f
