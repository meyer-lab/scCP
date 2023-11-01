import numpy as np
from .common import subplotLabel, getSetup, openPf2
from ..gating import gateThomsonCells
from ..parafac2 import pf2
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

    ax, f = getSetup((16,12), (2, 2))
    subplotLabel(ax)

    # data = data.to_memory(copy=True)
    sampled_data = data[(data.obs['Cell Type'] != 'Monocytes') | (data.obs['Condition'] != 'CTRL4')]

    # sampled_data = data.to_memory(copy=True)
    # for drug in data.obs.Condition.unique():
    #     for cell_type in data.obs['Cell Type'].unique():
    #         filt = (sampled_data.obs['Cell Type'] != cell_type) | (
    #                 sampled_data.obs.Condition != drug
    #             )
    #         filt[
    #                 np.random.choice(filt[~filt].index, int(len(filt[~filt].index) * 0.99), replace=False)
    #             ] = True
    #         sampled_data = sampled_data[filt]

    sampledX = pf2(sampled_data, "Condition", rank)
    factors = [sampledX.uns["Pf2_A"], sampledX.uns["Pf2_B"], sampledX.varm["Pf2_C"]]

    plotFactors(factors, sampledX, ax[0:3], reorder=(0, 2), trim=(2,))

    return f
