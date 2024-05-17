"""
Cytokine: Highly weighted genes per component
"""

from .common import getSetup
from .commonFuncs.plotFactors import plot_gene_factors_partial
from ..imports import import_cytokine
from ..factorization import correct_conditions
from ..imports import import_cytokine
from ..factorization import pf2
import pandas as pd

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((21, 24), (10, 8))
    
    X = import_cytokine()
    X = pf2(X, 40, tolerance=1e-6) 

    X.uns["Pf2_A"] = correct_conditions(X) 
    '''
    for i in range(X.uns["Pf2_A"].shape[1]):
        plot_gene_factors_partial(i + 1, X, ax[2 * i], geneAmount=50, top=True)
        plot_gene_factors_partial(i + 1, X, ax[(2 * i) + 1], geneAmount=50, top=False)
   
    '''
    relevant_components = []  # To store components containing the specified genes
    
    for i in range(X.uns["Pf2_A"].shape[1]):
        df = pd.DataFrame(data=X.varm["Pf2_C"][:, i], index=X.var_names, columns=[f"Cmp. {i+1}"])
        top_50_genes = df.sort_values(by=[f"Cmp. {i+1}"], ascending=False).index[:10]
        bot_50_genes = df.sort_values(by=[f"Cmp. {i+1}"], ascending=True).index[:10]
        
        
        plot_gene_factors_partial(i + 1, X, ax[2 * i], geneAmount=10, top=True)
        plot_gene_factors_partial(i + 1, X, ax[(2 * i) + 1], geneAmount=10, top=False)
        count = sum(1 for gene in top_50_genes if gene in ["CTLA4", "TGFB1", "EBI3", "IL-10", "CD73", "TIGIT", "CD39", "LAG3", "TIM3", "TIM1", "FOXP3", "TGFB"])
        count = count + sum(1 for gene in bot_50_genes if gene in ["CTLA4", "TGFB1", "EBI3", "IL-10", "CD73", "TIGIT", "CD39", "LAG3", "TIM3", "TIM1", "FOXP3", "TGFB"])
        if count >= 2:
            relevant_components.append(i + 1)
        
    # 1, 4, 5, 6, 7, 17, 19, 22, 27
    print("Components containing 2 or more specified genes in their top 10:", relevant_components)
  

    return f
