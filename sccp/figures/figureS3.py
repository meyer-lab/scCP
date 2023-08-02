"""
S3: Logisitic Regression (and maybe SVM) on Pf2 Factor matrix A output
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: Test various Pf2 ranks to see which best predicts disease status

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup,
    plotPf2RankTest
)
from ..imports.scRNA import load_lupus_data
from ..logisitcReg import testPf2Ranks
import matplotlib

# want to be able to see the different linetypes for this figure
matplotlib.rcParams["legend.handlelength"] = 2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), # fig size
                     (1, 1) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    lupus_tensor, _, group_labs = load_lupus_data(every_n=10) # leaving in every_n for the github checks
    

    ranks_to_test = [2, 3]
    penalties_to_test = [10, 50, 200, 1000]

    results = testPf2Ranks(lupus_tensor, group_labs, ranks_to_test, 
                           penalties_to_test=penalties_to_test)
    plotPf2RankTest(results, ax[0])

    return f