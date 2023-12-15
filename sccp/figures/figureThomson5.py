"""
Thomson: UMAP labeled by genes 
"""
import anndata
import numpy as np
from .common import subplotLabel, getSetup
from .commonFuncs.plotUMAP import plotGeneUMAP


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (1, 2))

    # Add subplot labels
    subplotLabel(ax)

    XX = anndata.read_h5ad("factor_cache/Lupus.h5ad", backed="r")

    from ..imports import import_lupus
    
    X = import_lupus()
    # cd4 = ["IL7R"]
    # cd8 = ["CD8A", "CD8B"]
    # nk = ["GNLY", "NKG7"]
    # mono1 = ["CD14", "LYZ", "MS4A7"]
    # mono2 = ["FCGR3A", "CST3"]
    # dc = ["CCR7", "HLA-DQA1", "GPR183"]
    # b = ["MS4A1", "CD79A"]
    # genes = np.concatenate((cd4, cd8, nk, mono1, mono2, dc, b))
    # genes = ["TUBB1", "PF4", "CLU", "PPBP", "HIST1H2AC", "ISG15", "LY6E", "IFI44L", "ANXA1", "IFI6"]
    genes = ["PPBP"]
    
    for i, gene in enumerate(genes):
        plotGeneUMAP(gene, "Pf2", X, XX, ax[i])

    return f
