"""
S2: Examining cell state for Pf2 on lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: visualize the component compostition by cell type

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


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of data
    _, cell_types, _ = load_lupus_data()  # don't need to get patient color mappings
    rank = 39
    #cmp = 7

    # WOULD ACTUALLY BE USING THE FOLLOWING:
    _, factors, projs, = openPf2(rank = rank, dataName = 'lupus', optProjs=True)


    proj_B = projs @ factors[1]

    comps_to_investigate = [14, 16, 17, 22, 26, 29]

    for i in range(len(comps_to_investigate)):
        cmp = comps_to_investigate[i]
        plotCompViolins(proj_B, cell_types, cmp, ax[i])

    return f
