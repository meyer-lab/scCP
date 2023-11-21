"""
Thomson: XX
"""
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
)
from .commonFuncs.plotUMAP import plotLabelsUMAP, plotCmpUMAP
from .commonFuncs.plotGeneral import (
    plotGenePerCellType,
    plotGenePerCategCond,
    plotGeneFactors,
    gene_plot_cells,
)
from ..gating import gateThomsonCells
from ..imports import import_thomson
from ..factorization import pf2

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((14, 14), (4, 4))

    # Add subplot labels
    subplotLabel(ax)
    rank = 30
    # X = openPf2(rank, "Thomson")
    X, cond = import_thomson(normalize=True)
    X = pf2(X, rank)
    # X = openPf2(rank, "Thomson")
    gateThomsonCells(X)
    
    X = X


    # cmp20 = ["UBE2C", "TOP2A", "TPX2", "ASPM", "BIRC5", "CXCL10", "GBP5", "ANKRD22", "FCGR1A", "ETV7"]

    cmp26 = ["MT1G", "MT1H", "MT1M", "NCCRP1", "FBN1", "SLC30A3", "VPREB3", "IL2RA", "LTA", "GINS2"]

    # cmp29 = ["GINS2", "KIAA0101", "TYMS", "BCAS4", "MS4A1", "CCL19","RSAD2", "IFIT1", "SIGLEC1", "CXCL11"]
    
    cmp29 = ["MS4A1"]



    print(X)
    plotGenePerCategCond(["CTRL6"], "CTRL6", cmp29, X, ax[0:len(cmp29)])


    return f
