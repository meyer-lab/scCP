from .common import getSetup
from ..imports import import_thomson
from ..factorization import pf2
from .commonFuncs.plotFactors import (
    plotConditionsFactors,
    plotCellState,
    plotGeneFactors,
)
from ..imports import import_thomson
from .figureThomson1 import groupDrugs
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from typing import Optional
from .commonFuncs.plotGeneral import (
    plotGenePerCellType,
)
import numpy as np
import seaborn as sns
import scipy.cluster.hierarchy as sch
from anndata import AnnData
from matplotlib.patches import Patch

cmap = sns.diverging_palette(240, 10, as_cmap=True)


def makeFigure():
    rank = 20
    data = import_thomson()

    sampled_data = data[
        (data.obs["Cell Type"] != "B Cells") | (data.obs["Condition"] != "CTRL4")
    ]

    ax, f = getSetup((10, 10), (1, 1))

    origX = pf2(data, rank, doEmbedding=False)

    sampledX = pf2(sampled_data, rank, doEmbedding=False)

    X = np.array(origX.uns["Pf2_A"])
    X = X[:, 9]
    Y = np.array(sampledX.uns["Pf2_A"])
    Y = Y[:, 9]

    yt = pd.Series(np.unique(origX.obs["Condition"]))
    yt2 = pd.Series(np.unique(sampledX.obs["Condition"]))

    assert yt.equals(yt2)

    ax[0].scatter(X, Y, s=1)
    for i, txt in enumerate(yt):
        ax[0].annotate(txt, (X[i], Y[i]), fontsize=8)
        
    ax[0].set_xlabel("Original Full Data")
    ax[0].set_ylabel("Sampled Data With B Cells Removed From CTRL4")

    return f
