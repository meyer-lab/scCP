"""
S2: Examining cell state for Pf2 on lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: visualize the component compostition by cell type
import pacmap
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotUMAP import plotCmpPerCellType, plotCmpUMAP, points
# from ..imports.scRNA import load_lupus_data
from ..parafac2 import pf2
import pandas as pd

import numpy as np

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 18), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    X = openPf2(rank, "Lupus")
    
    comps = [13, 14, 16, 26, 29, 32]
    cmp = 13
    plotCmpPerCellType(X, cmp, ax[0])
    


    # for i, comp in enumerate(comps):
    #     plotCmpPerCellType(weightedProjDF, comp, ax[(2*i)], outliers=False)
    #     plotCmpUMAP(comp, factors[1], pf2Points, projs, ax[(2*i)+1])


    return f
