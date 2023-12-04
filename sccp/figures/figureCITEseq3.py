"""
CITEseq: UMAP weighted by protein expression
"""
import numpy as np
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotUMAP import plotGeneUMAP


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("factor_cache/CITEseq.h5ad", backed="r")

    CSF = X.var_names.str.contains("Ace", case=False)
    print(X.var_names[CSF])

    print(X[:, "Ace"])

    # Csf1r
    

    names = X.var_names[X.var["feature_types"] == "Antibody Capture"]

    names = ["Apoe", "Hpgd", "Pltp", "Ms.CD11c"]

    print(np.nonzero(X[:, "Apoe"].varm["Pf2_C"] > 0.03))
    print(np.nonzero(X[:, "Hpgd"].varm["Pf2_C"] > 0.03))

    for i, name in enumerate(names[0:4]):
        plotGeneUMAP(name, "Pf2", X, ax[i])

    return f
