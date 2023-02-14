"""
Parafac2 implementation on PBMCs treated across treatments and patients
"""
import numpy as np
from .common import subplotLabel, getSetup, plotSCCP_factors, renamePlotsCoH
from ..imports.CoH import CoH_xarray
from ..parafac2 import parafac2_nd
from ..decomposition import plotR2X_CC


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 20), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Patient, Treatment, Cell, Marker]
    cohXA, celltypeXA = CoH_xarray(allmarkers=True, saveXA=False)
    print(np.unique(celltypeXA))

    # Shrink dataset
    cohXA = cohXA.loc[:, :, :50, :]
    celltypeXA = celltypeXA.loc[:, :, :50]

    # Performing parafac2 on single-cell Xarray
    rank = 3
    _, factors, projs = parafac2_nd(cohXA.to_numpy(), rank=rank, verbose=True) 

    plotSCCP_factors(factors, cohXA, projs[0, :3, :, :], ax, celltypeXA[0, :3, :], color_palette, plot_celltype=True)
    renamePlotsCoH(ax)
    
    plotR2X_CC(cohXA.to_numpy(), rank, ax[10], ax[11])


    return f

color_palette = [
    "black",
    "lightcoral",
    "red",
    "darksalmon"
    "darkorange",
    "peru",
    "tan"
    "yellow",
    "darkgoldenrod"
    "green",
    "turquoise",
    "blue",
    "blueviolet",
    "plum",
    "pink",
    "saddlebrown",
    "gold",
    "grey",
    "olive",
    "darkseagreen",
    "aqua",
    "fuchsia"
    "deeppink"
]
