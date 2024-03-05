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
from .figureCITEseq5 import top_bot_genes
from scipy.stats import linregress
from .figureLupus19 import cmpGatedDF
from .figureLupus17 import dfGenePerStatus, plotCmpPerGene


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 16), (5, 4))


    # Add subplot labels
    subplotLabel(ax)
    
    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    cmp=28
    genes = top_bot_genes(X, cmp=cmp, geneAmount=5)
    partial = cmpGatedDF(X, cmp, perc=5, positive=True)
        
    partial = partial[1]
    geneDF = dfGenePerStatus(partial, genes[0], cellType="Cell Type2")


    idx = len(np.unique(geneDF["Cell Type"]))
    plotCmpPerGene(partial, cmp, geneDF, ax[0:idx])
    

    return f

