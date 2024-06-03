"""
Lupus: Average gene expression stratified by cell type and status
"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
import pandas as pd
from .commonFuncs.plotFactors import bot_top_genes
from .commonFuncs.plotGeneral import plot_avegene_per_celltype
from matplotlib.axes import Axes
import anndata


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 15.5), (4, 5))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad", backed="r")

    genes = bot_top_genes(X, cmp=20, geneAmount=30)
    for i in genes[30:]:
    # for i in genes[:30]:
        print(i)

    # for i, gene in enumerate(np.ravel(genes)):
    #     plot_avegene_per_celltype(X, gene, ax[i], cellType="Cell Type2")

    return f

