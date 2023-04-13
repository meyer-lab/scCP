"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
import numpy as np
import multiprocessing as mp
import os
import pickle
import pandas as pd
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from .common import (
    subplotLabel,
    getSetup,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 25), (2, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    
    genes = 13409
    repeat = 10
    rank = 30
    
    geneFactors = np.empty([genes, rank, repeat])
    for i in range(repeat):
        data = ThompsonXA_SCGenes(shuffle="On") 
        _, factors, _, _ = parafac2_nd(
            data,
            rank=rank,
        )
        
        geneFactors[:, :, i] = factors[2]
        
    means = geneFactors.mean(axis=2)
    devs = geneFactors.std(axis=2)

    data = ThompsonXA_SCGenes() 
    _, factors, _, _ = parafac2_nd(
            data,
            rank=rank,
        )
    
    t_stats = (factors[2] - means) / devs
    over = np.clip(t_stats, a_min=0, a_max=None)
    under = -np.clip(t_stats, a_min=None, a_max=0)
    
    over_p = stats.norm.sf(np.abs(over))
    under_p = stats.norm.sf(np.abs(under))
    
    for col in np.arange(over_p.shape[1]):
        over_p[:, col] = multipletests(over_p[:, col])[1]
        under_p[:, col] = multipletests(under_p[:, col])[1]

    over_df = pd.DataFrame(
        over_p,
        index=data.variable_labels,
        columns=np.arange(1, rank + 1)
    )
    under_df = pd.DataFrame(
        under_p,
        index=data.variable_labels,
        columns=np.arange(1, rank + 1)
    )
    
    print(over_df)
    print(under_df)
   


    return f


