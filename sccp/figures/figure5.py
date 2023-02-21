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
    ax, f = getSetup((15, 15), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Patient, Treatment, Cell, Marker]
    cohXA, celltypeXA = CoH_xarray(saveXA=False)

    cohXA = cohXA.drop(
        labels=("SSC-H", "SSC-A", "FSC-H", "FSC-A", "SSC-B-H", "SSC-B-A", "Live/Dead"),
        dim="Marker",
    )

    # Normalize here
    cohXA.values -= np.nanmean(cohXA.values, axis=(0, 1, 2), keepdims=True)

    # Then finish off missing values with zero
    cohXA.values = np.nan_to_num(cohXA.values)

    # Shrink dataset
    cohXA = cohXA.loc[:, :, ::400, :]
    celltypeXA = celltypeXA.loc[:, :, ::400]

    # Performing parafac2 on single-cell Xarray
    rank = 8
    _, factors, projs, _, _ = parafac2_nd(cohXA.to_numpy(), rank=rank, verbose=True)

    plotSCCP_factors(
        factors,
        cohXA,
        projs[0, :2, :, :],
        ax,
        celltypeXA[0, :2, :],
        color_palette,
    )
    renamePlotsCoH(ax)

    plotR2X_CC(cohXA.to_numpy(), rank, ax[9], ax[10])

    return f


color_palette = [
    "black",
    "lightcoral",
    "red",
    "darksalmon",
    "darkorange",
    "peru",
    "tan",
    "yellow",
    "darkgoldenrod",
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
    "fuchsia",
    "deeppink",
]
