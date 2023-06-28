"""Factors from scib paper"""
import numpy as np
from .common import subplotLabel, getSetup, plotFactors, plotProj, plotR2X, plotCV
from ..imports.scib import import_scib_data
from ..parafac2 import parafac2_nd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 10), (2, 3))

    # Add subplot labels
    subplotLabel(ax)
   
    data = import_scib_data(dataname="Stimulation1")
    data = import_scib_data(dataname="ImmuneHuman")
    data = import_scib_data(dataname="Stimulation2")
    data = import_scib_data(dataname="ImmuneHumanMouse")
    data = import_scib_data(dataname="Pancreas")

    rank = 2

    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
    )

    plotFactors(factors, data, ax[0:3], reorder=(0, 2), trim=(2,))
    
    return f