"""
S3d: Plot samples along two components to see patient separation
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: see if SLE/healthy samples can be stratified along strongly predictive Pf2 components
# (they can, at least when you do 13 and 26)

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup,
    openPf2
)
from ..imports.scRNA import load_lupus_data

import numpy as np
import pandas as pd
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
    group_to_predict = "SLE_status" # group to predict 

    lupus_tensor, obs = load_lupus_data() # leaving in every_n for the github checks

    group_labs = (
            obs[['sample_ID', group_to_predict]]
            .drop_duplicates()
        )

    group_labs = group_labs.set_index('sample_ID')
    print(group_labs)


    _, factors, _, = openPf2(rank = rank, dataName = 'lupus', optProjs=True)

    factor_A = pd.DataFrame(factors[0],
                            columns = [f"Cmp. {i}" for i in np.arange(1, rank + 1)],
                            index = lupus_tensor.condition_labels)
    print(factor_A)

    merged = factor_A.merge(group_labs, left_index=True, right_index=True)
    print(merged)

    sns.scatterplot(data = merged, x = 'Cmp. 13', y = 'Cmp. 26', hue = 'SLE_status')

    return f