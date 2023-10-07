"""
Hamad CITEseq dataset
"""
from .common import subplotLabel, getSetup
from .commonFuncs.plotFactors import (
    plotFactors,
)
from ..imports.citeseq import import_citeseq
from .commonFuncs.plotUMAP import points
from ..parafac2 import pf2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((25, 25), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = import_citeseq()
    X = pf2(X, "Condition", rank=40)

    plotFactors(factors, data, ax[0:3], reorder=(0, 2), trim=(2,))

    points(
        pf2Points,
        labels=protDF["Condition"].values,
        ax=ax[3],
        show_legend=True,
    )

    return f
