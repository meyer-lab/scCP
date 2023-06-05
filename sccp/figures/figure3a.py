"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
import umap
from .common import subplotLabel, getSetup, umap_axis
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 8), (2, 2))
    subplotLabel(ax)  # Add subplot labels

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    rank = 25
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
        verbose=True,
    )
    allP = np.concatenate(projs, axis=0)
    CompW = allP @ factors[1]

    # UMAP dimension reduction
    ump = umap.UMAP(random_state=1).fit_transform(allP)
    x = ump[::10, 0]
    y = ump[::10, 1]


    rank=25
    tl = ax[0].scatter(
        x,
        y,
        marker=".",
        c=CompW[::10, rank-1],
        cmap="PRGn",
        vmin=-0.3,
        vmax=0.3,
        linewidths=0,
        s=15.0,
    )
    f.colorbar(tl, ax=ax[0])
    ax[0].set(title=f"Component {rank}")
    umap_axis(x, y, ax[0])
    
    rank=23
    tl = ax[1].scatter(
        x,
        y,
        marker=".",
        c=CompW[::10, rank-1],
        cmap="PRGn",
        vmin=-0.3,
        vmax=0.3,
        linewidths=0,
        s=15.0,
    )
    f.colorbar(tl, ax=ax[1])
    ax[1].set(title=f"Component {rank}")
    umap_axis(x, y, ax[1])

    return f
