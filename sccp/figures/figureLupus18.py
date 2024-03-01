"""
Thomson: Plotting normalized genes and separating data by status (and celltype)
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
from .commonFuncs.plotUMAP import plotCmpGeneWeightedUMAP, plotCmpGeneWeightedPerCellType, plot2CmpUMAP

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 18), (6, 4))
    # ax, f = getSetup((12, 6), (2, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    # print(X)
    x = X.obsm["weighted_projections"][:, 0]
    
    
    idx = X.obs.reset_index(drop=True).index.values
    s = np.concatenate(([X.obsm["weighted_projections"][:, 0]], [X.obsm["weighted_projections"][:, 1]], [x]))
    print(np.shape(s))
    
#     pri
    
    threshold = np.percentile(s[0,:], 10) # calculate the 10th percentile
    print(threshold)
    newx = s[:, s[0,:] < threshold]
    
    print(newx)
    print(np.shape(newx))

#    X = X[ind[:, cmp-1], :]
   
    


    return f



