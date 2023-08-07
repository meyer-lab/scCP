"""
S4: what's going on in component 11
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: figure out whats going on in component 11

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup,
    openPf2
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

    lupus_tensor, _, _ = load_lupus_data() 

    genes = lupus_tensor.variable_labels
    
    _, factors, _, = openPf2(rank = rank, dataName = 'lupus', optProjs=True)

    C_matrix = pd.DataFrame(factors[2], columns = [f"comp_{i}" for i in np.arange(1, rank + 1)], index = genes)

    # threshold of 0.025 decided by looking at threshold of 0.01; then chose elbow
    high_expr11 = C_matrix[C_matrix['comp_11'] > 0.025]['comp_11'].sort_values(ascending=False)

    #high_expr16 = expr11[expr11['comp_11'] > 0.01]
    print(high_expr11)
    print(high_expr11.describe())

    bar_data = pd.DataFrame(high_expr11).reset_index()
    print(bar_data)
    sns.barplot(data = bar_data, x = 'index', y = 'comp_11', ax = ax[0])
    
    ax[0].tick_params(axis="x", rotation=90)



    return f