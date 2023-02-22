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
    ax, f = getSetup((20, 20), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Patient, Treatment, Cell, Marker]
    cohXA, celltypeXA = CoH_xarray(saveXA=False)
    print(np.unique(celltypeXA))
    # print(cohXA)
    cohXA = cohXA.loc[:, :, :, marker_surface_stat]
    
    # print(np.shape(np.nanmean(cohXA.values, axis=(0,1, 2), keepdims=True)))
    # print(np.shape(np.nanmean(cohXA.values, axis=(0,1), keepdims=True)))
    # print(np.shape(np.nanmean(cohXA.values, axis=(0, 2), keepdims=True)))
    
    
    cohXA.values /= np.nanmean(cohXA.values, axis=(0, 1, 2), keepdims=True)
    cohXA.values /= np.nanstd(cohXA.values, axis=(0, 1, 2), keepdims=True)
    # print(cohXA)
    
    # # Normalize here
    # cohXA.values /= np.nanmean(cohXA.values, axis=(0, 1), keepdims=True)

    # # # Then finish off missing values with zero
    cohXA.values = np.nan_to_num(cohXA.values)
    
    # # # # Shrink dataset
    cohXA = cohXA.loc[:, :, ::25, :]
    celltypeXA = celltypeXA.loc[:, :, ::25]

    # # # Performing parafac2 on single-cell Xarray
    rank = 4
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

]



marker_surface_stat = [
    "pSTAT6", 
    "pSTAT3", 
    "pSTAT1", 
    "pSmad1-2", 
    "pSTAT5", 
    "pSTAT4",
    "CD45RA",
    "CD4",
    "CD16",
    "CD8",
    "PD-L1",
    "CD3",
    "PD-1",
    "CD14",
    "CD33",
    "CD27",
    "FoxP3",
    "CD20",
]
    
    
cell_types = [
    "CD16 NK",
    "CD8+",
    "CD4+",
    "CD4-/CD8-",
    "Treg",
    "CD20 B",
    "Classical Monocyte",
    "NC Monocyte"]