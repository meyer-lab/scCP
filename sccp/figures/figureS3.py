"""
S3: Logisitic Regression (and maybe SVM) on Pf2 Factor matrix A output
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: run logisitc regression to see which components are best able to predict disease status

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup,
    plotFactors,
    plotWeight
)
from ..parafac2 import parafac2_nd
from ..imports.scRNA import load_lupus_data
from .common import subplotLabel, getSetup
import pandas as pd
import numpy as np



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), # fig size
                     (2, 2) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    rank = 30

    lupus_tensor, _, row_colors = load_lupus_data(every_n = 10) # don't need to grab cell types here

    _, factors, _, _ = parafac2_nd(lupus_tensor, 
                                    rank = rank, 
                                    n_iter_max= 20,
                                    random_state = 1, 
                                    verbose=True)
    
    patients = lupus_tensor.condition_labels

    A_matrix = pd.DataFrame(factors[0], 
                            index = patients,
                            columns = [f"comp_{i}" for i in np.arange(1, rank + 1)])
    
    print(A_matrix)

    assert False

    plotFactors(factors, lupus_tensor, ax[0:3], reorder = (0,2), trim=(2,), row_colors = row_colors)

    plotWeight(weights, ax[3])
    ax[3].set_title('Weight of Each Component')

    return f