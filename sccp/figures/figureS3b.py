"""
S3b: Logistic Regression (and maybe SVM) on Pf2 Factor matrix A output
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: run logisitc regression to see which components are best able to predict disease status

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
    plotCmpRegContributions
)
from ..imports.scRNA import load_lupus_data
from ..logisticReg import getCompContribs




def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 6), # fig size
                     (1, 1) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    rank = 40

    _, obs = load_lupus_data() # leaving in every_n for the github checks

    _, factors, _, = openPf2(rank = rank, dataName = 'lupus', optProjs=True)
        
    A_matrix = factors[0]

    groups_to_predict = {'SLE_status': 'SLE',
                         'ancestry': 'European'}

    contribs = getCompContribs(A_matrix, obs, groups_to_predict, penalty_amt= 50)
        
    plotCmpRegContributions(contribs, 'target_variable', ax[0])

    return f