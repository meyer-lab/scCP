"""
S4c: Plot Top and Bottom expressed Genes in a Component
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: look at genes and their weights, because over enrichment analysis doesn't account for weight
# maybe could be extrapolated to be 

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
    ax, f = getSetup((15, 5), # fig size
                     (1, 3) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    component = 32
    top_n = 40
    # which GO term, enriched in the top_n genes weighted; do you want to plot for...
    # GO terms from the top weighted genes?
    top_term = 1
    # GO terms from the bottom weighted genes?
    bottom_term = 1
    # what GO set do you want to use?
    geneset = 'GO_Biological_Process_2021'

    lupus_tensor, _ = load_lupus_data() 

    genes = lupus_tensor.variable_labels
    
    _, factors, _ = openPf2(rank = rank, dataName = 'lupus', optProjs=True)

    C_matrix = pd.DataFrame(factors[2], columns = [f"comp_{i}" for i in np.arange(1, rank + 1)], index = genes)

    # can get GO terms to try using overenrichment in the top n weighted genes (top is high positive weight; bot is high negative weight)
    top_go, bottom_go = getGOFromTopGenes(C_matrix, component, top_n=top_n, geneset=geneset)
    
    # it makes sense to print these out to see if we want to use different terms from here in future runs
    # using the printed out version would be how you choose to change `top_term` or `bottom_term` defined above
    print("\n GO TERMS FROM THE HIGHEST POSITIVELY WEIGHTED GENES\n", top_go.head(15))
    print("\n GO TERMS FROM THE HIGHEST NEGATIVELY WEIGHTED GENES\n", bottom_go.head(15))
    # get the actual string ('GO:#######') that can be passed to `plotGenesFromGO`
    go_term_top = top_go.index[top_term - 1]
    go_term_bottom = bottom_go.index[bottom_term - 1]


    plotGenesFromGO(go_term_top, C_matrix, component, ax[0])
    plotGenesFromGO(go_term_bottom, C_matrix, component, ax[1])
    # example where we give an accession number specifically:
    plotGenesFromGO('GO:0060337', C_matrix, 32, ax[2], accession=True)

    return f