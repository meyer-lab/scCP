"""
Hamad CITEseq dataset
"""
from .common import (
    subplotLabel,
    getSetup,
)
from ..imports.citeseq import import_citeseq
from .commonFuncs.plotUMAP import plotGeneUMAP
from ..parafac2 import pf2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((50, 50), (5, 5))

    # Add subplot labels
    subplotLabel(ax)

    X = import_citeseq()
    X = pf2(X, "Condition", rank=3)

    names = X.var_names[X.var["feature_types"] == "Antibody Capture"]

    plotGeneUMAP(names, "Pf2", X.obsm["umap"], X, ax[0:25])

    return f
