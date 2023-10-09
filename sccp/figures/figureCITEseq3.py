"""
Hamad CITEseq dataset
"""
import numpy as np
import pacmap

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

    pf2Points = pacmap.PaCMAP().fit_transform(projs)

    protNames = np.unique(protDF.drop(columns="Condition").columns)

    # protNames = protNames[0:24]
    # protNames = protNames[24:48]
    # protNames = protNames[50:75]
    # protNames = protNames[75:100]
    protNames = protNames[100:].tolist()

    plotGeneUMAP(protNames, "Pf2", pf2Points, protDF, ax[0:25])

    return f
