"""
S3: Logisitic Regression (and maybe SVM) on Pf2 Factor matrix A output
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: run logisitc regression and assess predictive ability of 39 comp Pf2 using ROC AUC
# comparison to Perez et al (linked above) fig 4C

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
    plotPf2ROC
)
from ..imports.scRNA import load_lupus_data

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 6), # fig size
                     (1, 1) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    rank = 39

    lupus_tensor, _, group_labs = load_lupus_data(give_batch=True) 

    patients = lupus_tensor.condition_labels
    
    _, factors, _, = openPf2(rank = rank, dataName = 'lupus')

    A_matrix = factors[0]

    penalties_to_test = [10, 20, 30, 50, 100, 150, 200, 1000]

    plotPf2ROC(A_matrix, patients, group_labs, rank, ax[0], penalties_to_test= penalties_to_test)


    return f