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

from tlviz.factor_tools import factor_match_score as fms
from tensorly.cp_tensor import cp_flip_sign, CPTensor


def makeFigure():
    rank = 30
    data = import_thomson()
    gateThomsonCells(data)

    ax, f = getSetup((16,12), (2,2))
    subplotLabel(ax)

    #figures for all dataset
    dataX = pf2(data, rank, random_state=1)
    # plotLabelsUMAP(dataX, "Cell Type", ax[0])
    factors = [dataX.uns["Pf2_A"], dataX.uns["Pf2_B"], dataX.varm["Pf2_C"]]
    # plotFactors(factors, dataX, ax[1:4], reorder=(0, 2), trim=(2,))

    #loop to drop data
    sampled_data = data
    #convert the index to string
    sampled_data.obs.index = data.obs.index.astype(str)

    for drug in data.obs.Condition.unique():
        for cell_type in data.obs["Cell Type"].unique():
            filt = (
                sampled_data.obs['Cell Type'] != cell_type ) | (
                sampled_data.obs.Condition != drug
            )
        
        idx_to_keep = np.random.choice(
            filt[~filt].index.astype(str),
            int(len(filt[~filt].index) * 0.95),
            replace=False
        )
        filt[idx_to_keep] = True
        sampled_data = sampled_data[filt]


    #figures for sampled datasets
    sampledX = pf2(sampled_data, rank, random_state=3)
    plotLabelsUMAP(sampledX, "Cell Type", ax[0])
    factors = [sampledX.uns["Pf2_A"], sampledX.uns["Pf2_B"], sampledX.varm["Pf2_C"]]
    plotFactors(factors, sampledX, ax[1:4], reorder=(0, 2), trim=(2,))

    #factor score match
    dataXcp = CPTensor((dataX.uns["Pf2_weights"], [dataX.uns["Pf2_A"], dataX.uns["Pf2_B"], dataX.varm["Pf2_C"]]))
    sampledXcp = CPTensor((sampledX.uns["Pf2_weights"], [sampledX.uns["Pf2_A"], sampledX.uns["Pf2_B"], sampledX.varm["Pf2_C"]]))

    fmsScore = fms(dataXcp, sampledXcp, consider_weights=True, skip_mode=None)
    print(fmsScore)

    return f
