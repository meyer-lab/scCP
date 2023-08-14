"""
S4: Investigation of Component 13
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: look at the cells that are highly contributing to component 13; see if they're megakaryocytes

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

def investigate13(obs_column, ax, threshold = 0.05):
    rank = 40
    #obs_column = 'cell_type_broad'

    _, obs = load_lupus_data() 

    ct = obs[obs_column]

    
    _, factors, projs = openPf2(rank = rank, dataName = 'lupus', optProjs=True)

    proj_B = projs @ factors[1]

    proj_B = pd.DataFrame(proj_B,
                 index = obs.index,
                 columns = [f"comp_{i}" for i in np.arange(1, rank + 1)])
    
    proj_et_obs = proj_B.merge(ct, left_index=True, right_index=True)
    cmp_13 = proj_et_obs[[obs_column, 'comp_13']]
    # get just the ones that are "super" positive
    counts_all = cmp_13.groupby(by = obs_column).count().reset_index().rename({'comp_13':'count'}, axis = 1)
    cmp_13 = cmp_13[cmp_13['comp_13'] > threshold]

    counts = cmp_13.groupby(by = obs_column).count().reset_index().rename({'comp_13':'count'}, axis = 1)
    print(counts['count'])
    print(counts_all['count'])
    print(counts['count']/counts_all['count'])

    pcts = pd.concat((counts[obs_column], counts['count']/counts_all['count']), axis = 1).rename({'count': 'percent'}, axis = 1)
    pcts['percent'] = pcts['percent'] * 100
    print(pcts)

    sns.barplot(pcts, x = obs_column, y = 'percent', errorbar=None, ax=ax)
    ax.tick_params(axis="x", rotation=90)
    ax.set_title(obs_column + ' Percentages, Threshold: ' + str(threshold))



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 8), # fig size
                     (3, 2) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    investigate13('cell_type_broad', ax[0], threshold=0.1)
    investigate13('louvain', ax[1], threshold=0.1)
    investigate13('cell_type_broad', ax[2])
    investigate13('louvain', ax[3])
    investigate13('cell_type_broad', ax[4], threshold=0.0)
    investigate13('louvain', ax[5], threshold=0.0)

    return f