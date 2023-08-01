"""
S2: Examining cell state for Pf2 on lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: visualize the cell state compostition by cell type/UMAP

from .common import (
    subplotLabel,
    getSetup,
    plotCmpUMAP,
    plotCompViolins,
    plotUMAP_ct,
    openPf2
)
from ..imports.scRNA import load_lupus_data
import umap
from sklearn.decomposition import PCA
# JUST SO THAT GITHUB CHECK WILL PASS-- USING ACTUAL PF2 NOT OPEN PF2
# (because projs file is too large to upload to github)
from parafac2 import parafac2_nd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of data
    _, cell_types, _ = load_lupus_data()  # don't need to get patient color mappings
    rank = 39
    cellState = 11
    cmp = 11

    _, factors, projs, = openPf2(rank = rank, dataName = 'lupus', optProjs=True)


    proj_B = projs @ factors[1]

    # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1).fit(projs)


    plotUMAP_ct(cell_types, pf2Points, ax[0])
    plotCmpUMAP(cellState, cmp, factors, pf2Points, projs, ax[1])
    plotCompViolins(proj_B, cell_types, cmp, ax[2])

    return f
