import numpy as np
from .common import subplotLabel, getSetup, openPf2
from ..gating import gateThomsonCells
from ..parafac2 import pf2
from .commonFuncs.plotGeneral import plotGeneFactors
from .commonFuncs.plotUMAP import (
    plotLabelsUMAP,
)
from .commonFuncs.plotFactors import (
    plotFactors,
)
from ..imports import import_thomson
from ..imports import import_thomson


def makeFigure():
    rank = 30
    data = import_thomson()
    gateThomsonCells(data)

    ax, f = getSetup((24, 24), (8, 8))

    # data = data.to_memory(copy=True)

    sampled_data = data[
        (data.obs["Cell Type"] != "T Cells") | (data.obs["Condition"] != "CTRL4")
    ]

    sampled_data = data

    sampledX = pf2(sampled_data, rank)
    factors = [sampledX.uns["Pf2_A"], sampledX.uns["Pf2_B"], sampledX.varm["Pf2_C"]]

    plotFactors(factors, sampledX, ax[0:3], reorder=(0, 2), trim=(2,))
    plotLabelsUMAP(sampledX, "Cell Type", ax[3])

    for i in np.arange(0, rank):
        plotGeneFactors(i + 1, sampledX, ax[2 * i + 4 : 2 * i + 6], geneAmount=5)

    return f
