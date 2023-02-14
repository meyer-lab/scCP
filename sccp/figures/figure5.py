"""
Parafac2 implementation on PBMCs treated across treatments and patients
"""
import numpy as np
from .common import subplotLabel, getSetup, plotSCCP_factors, renamePlotIL2
from ..imports.CoH import CoH_xarray
from ..parafac2 import parafac2_nd
from ..decomposition import plotR2X_CC


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Ligand, Dose, Time, Cell, Marker]
    # cohXA, celltypeXA = CoH_xarray(allmarkers=True, saveXA=False)

    # Shrink dataset
    # flowXA = flowXA.loc[:, :, :, :500, :]
    # celltypeXA = celltypeXA.loc[:, :, :, :500]

    # # Performing parafac2 on single-cell Xarray
    # rank = 3
    # _, factors, projs = parafac2_nd(flowXA.to_numpy(), rank=rank, verbose=True) 

    # plotSCCP_factors(factors, flowXA, projs[7:9, 0, 0, :, :], ax, celltypeXA[7:9, 0, 0, :], color_palette, plot_celltype=True)
    # renamePlotIL2(ax)
    
    # plotR2X_CC(flowXA.to_numpy(), rank, ax[10], ax[11])


    return f

color_palette = ["turquoise", "blueviolet", "green"]