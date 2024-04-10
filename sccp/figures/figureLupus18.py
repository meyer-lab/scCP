"""
Lupus
"""
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr
import scanpy as sc

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 16), (5, 4))


    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    sc.tl.rank_genes_groups(adata=X, groupby="leiden", method="wilcoxon")
    print(X)
    sc.pl.rank_genes_groups(X, n_genes=30, sharey=False, save=True)

    



    return f


