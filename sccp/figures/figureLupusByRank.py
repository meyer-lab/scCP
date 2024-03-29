"""
Lupus: Cross validation for determining optimal paramaters for logistic regression
Lupus is spelled wrong in the figure name to indicate that this figure does not depend
on cached results.
"""
import numpy as np
from ..imports import import_lupus
from ..logisticReg import testPf2Ranks
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotLupus import plotPf2RankTest
from .commonFuncs.plotLupus import getSamplesObs


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((4, 4), (1, 1))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    X = import_lupus()

    condStatus = getSamplesObs(X.obs)

    rank = list(np.arange(5, 11, 5))
    results = testPf2Ranks(
        X,
        condStatus,
        rank,
        cv_group="Processing_Cohort",
    )

    plotPf2RankTest(results, ax[0])
    ax[0].set(xlim=(0, np.amax(rank)), ylim=(0, 1))

    return f
