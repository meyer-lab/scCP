"""
S4c: Plot Top and Bottom expressed Genes in a Component
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: look at genes and their weights, because over enrichment analysis doesn't account for weight

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
    plotTopGenes
)
from ..imports.scRNA import load_lupus_data
import pandas as pd
import numpy as np
import seaborn as sns


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 6), # fig size
                     (1, 1) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    component = 13

    lupus_tensor, _ = load_lupus_data() 

    genes = lupus_tensor.variable_labels
    
    _, factors, _ = openPf2(rank = rank, dataName = 'lupus', optProjs=True)

    C_matrix = pd.DataFrame(factors[2], columns = [f"comp_{i}" for i in np.arange(1, rank + 1)], index = genes)

    plotTopGenes(C_matrix, component, ax[0], top_n = 25, top_term = 1, bottom_term = 2, verbose = True)

    return f