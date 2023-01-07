"""
Parafac2 implementation on PBMCs treated across IL2 treatments, times, and doses
"""
from .common import subplotLabel, getSetup, plotSCCP_factors
from ..imports.cytok import IL2_flowXA
from ..parafac2 import parafac2

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Gene, Cell, Drug]
    flowXA, _ = IL2_flowXA()

    # Performing parafac2 on single-cell Xarray
    rank = 5
    weights, factors, _ = parafac2(
        FlowXA.to_numpy(),
        rank=rank,
        tol=1e-8,
        nn_modes=(0, 1, 2),
        verbose=True
    )

    plotSCCP_factors(rank, factors, FlowXA, ax)

    return f
