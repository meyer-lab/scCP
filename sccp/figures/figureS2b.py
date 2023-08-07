"""
S2: Examining cell state for Pf2 on lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: visualize the component compostition by cell type

from .common import (
    subplotLabel,
    getSetup,
    plotCompViolins,
    openPf2
)
from ..imports.scRNA import load_lupus_data


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of data
    _, cell_types, _ = load_lupus_data()  
    rank = 40

    _, factors, projs, = openPf2(rank = rank, dataName = 'lupus', optProjs=True)


    proj_B = projs @ factors[1]

    comps_to_investigate = [13, 16, 26, 29]

    for i in range(len(comps_to_investigate)):
        cmp = comps_to_investigate[i]
        plotCompViolins(proj_B, cell_types, cmp, ax[i])

    return f
