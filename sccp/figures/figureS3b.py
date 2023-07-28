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
    openPf2
)
from ..imports.scRNA import load_lupus_data
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import matplotlib

# want to be able to see the different linetypes for this figure
matplotlib.rcParams["legend.handlelength"] = 2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 6), # fig size
                     (1, 1) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    rank = 39

    lupus_tensor, _, group_labs = load_lupus_data() # don't need to grab cell types here
    

    _, factors, _, = openPf2(rank = 39, dataName = 'lupus')
        
    A_matrix = factors[0]
        
        # train a logisitic regression model on that rank, using cross validation

    log_reg = LogisticRegression(random_state=0, max_iter = 5000, penalty = 'l1', solver = 'saga', C = 50)

    log_fit = log_reg.fit(A_matrix, group_labs.to_numpy())

    coefs = pd.DataFrame(log_fit.densify().coef_,
                         columns = [f"comp_{i}" for i in np.arange(1, rank + 1)]).melt(var_name = "Component",
                                                                                       value_name = "Weight")
    
    sns.barplot(data = coefs, x = "Component", y = "Weight", color = '#1a759f', ax = ax[0])
    ax[0].tick_params(axis="x", rotation=90)
    ax[0].set_title('Weight of Each component in Logsitic Regression')
    print(coefs)
    


    return f