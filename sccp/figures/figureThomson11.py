"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
)
from ..imports.gating import gateThomsonCells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    rank = 30
    X = openPf2(rank, "Thomson")

    X.obs["Cell Type"] = gateThomsonCells(X)

    genes = [
        "VPREB3",
        "CD79A",
        "FAM111B",
        "HOPX",
        "SLC30A3",
        "MS4A1",
        "CCNB1",
        "GADD45A",
        "SLC40A1",
        "CDC20",
        "ITIH3",
        "CPEB1",
    ]

    Wmat = X.obsm["weighted_projections"][:, 19]  # Get component 20

    for i, gene in enumerate(genes):
        df = pd.DataFrame(
            {
                gene: np.squeeze(X[:, gene].X),
                "Component Weight": Wmat,
                "Cell Type": X.obs["Cell Type"],
            }
        )

        sns.scatterplot(
            data=df, x="Component Weight", y=gene, ax=ax[i], hue="Cell Type"
        )

    return f
