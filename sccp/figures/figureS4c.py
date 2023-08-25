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
    plotGenesFromGO
)
from ..imports.scRNA import load_lupus_data
import pandas as pd
import numpy as np
from ..geneontology import getGOFromTopGenes


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 6), # fig size
                     (1, 2) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    component = 13
    top_n = 25
    top_term = 1
    bottom_term = 1

    lupus_tensor, _ = load_lupus_data() 

    genes = lupus_tensor.variable_labels
    
    _, factors, _ = openPf2(rank = rank, dataName = 'lupus', optProjs=True)

    C_matrix = pd.DataFrame(factors[2], columns = [f"comp_{i}" for i in np.arange(1, rank + 1)], index = genes)

    # can get GO terms to try using overenrichment in the top n weighted genes (top is high positive weight; bot is high negative weight)
    top_go, bottom_go = getGOFromTopGenes(C_matrix, component, top_n=top_n, geneset='GO_Cellular_Component_2023')
    
    # it makes sense to print these out to see if we want to use different terms from here in future runs
    print("\n GO TERMS FROM THE HIGHEST POSITIVELY WEIGHTED GENES\n", top_go.head(15))
    print("\n GO TERMS FROM THE HIGHEST NEGATIVELY WEIGHTED GENES\n", bottom_go.head(15))
    # get the actual string that can be passed to `plotGenesFromGO`
    go_term_top = top_go.index[top_term - 1]
    go_term_bottom = bottom_go.index[bottom_term - 1]


    plotGenesFromGO(go_term_top, C_matrix, component, ax[0])
    plotGenesFromGO(go_term_bottom, C_matrix, component, ax[1])
    # WE COULD ALSO: just give `plotGenesFromGO` an accession number, e.g.
    #go_ac = 'GO:0031093' and accession = True gives platelet alpha granule lumen genes 
    # (if we already know what we want)

    return f