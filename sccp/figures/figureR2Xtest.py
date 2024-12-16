"""
Figure Test
"""

import anndata
import pandas as pd
from .common import getSetup, subplotLabel
from ..factorization import pf2_pca_r2x


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (3, 3))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/pf2/thomson_fitted.h5ad")
    X.obs["condition_unique_idxs"] = pd.Categorical(X.obs["condition_unique_idxs"])
    ranks = [1, 2, 3, 4, 5]
    
    pf2_pca_r2x(X, ranks)
    
        
    return f
