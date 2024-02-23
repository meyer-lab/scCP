"""
Lupus: Plot 2 Pf2 factors for conditions
"""
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotLupus import getSamplesObs
import seaborn as sns
import numpy as np
import pandas as pd
from ..factorization import correct_conditions


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 10), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    predict = "SLE_status"
    condStatus = getSamplesObs(X.obs)
    condStatus = condStatus.set_index("Condition")

    df = pd.DataFrame(
        correct_conditions(X),
        columns=[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)],
        index=condStatus.index,
    )
    df = df.merge(condStatus, left_index=True, right_index=True)

    twoCmp = []
    for i in range(1,30,2):
        twoCmp.append([i, i+1])
        
    print(twoCmp)
    

    for i, pair in enumerate(twoCmp):
        sns.scatterplot(
            data=df, x=f"Cmp. {pair[0]}", y=f"Cmp. {pair[1]}", hue=predict, ax=ax[i]
        )

    return f
