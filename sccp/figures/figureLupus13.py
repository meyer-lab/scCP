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
    ax, f = getSetup((12, 10), (3, 4))

    X = read_h5ad("factor_cache/Lupus.h5ad", backed="r")

    comps = [43, 22, 30, 39, 48]
    for i,cmp in enumerate(comps):
        plotGeneFactors(cmp, X, ax[2 * i], geneAmount=10, top=True)
        plotGeneFactors(cmp, X, ax[2 * i + 1], geneAmount=10, top=False)

    return f
