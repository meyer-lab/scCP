import numpy as np
from .common import subplotLabel, getSetup, openPf2
from ..imports.gating import gateThomsonCells
from ..parafac2 import pf2
from .commonFuncs.plotUMAP import (
    plotLabelsUMAP,
)


def makeFigure():
    rank = 30
    data = openPf2(rank, "Thomson")
    data.obs["cell_type"] = gateThomsonCells(data)

    ax, f = getSetup((10,7), (1, 1))
    subplotLabel(ax)

    sampled_data = data.copy()
    for drug in data.obs.Condition.unique():
        for cell_type in data.obs.cell_type.unique():
            filt = (sampled_data.obs.cell_type != cell_type) | (
                    sampled_data.obs.Condition != drug
                )
            filt[
                    np.random.choice(filt[~filt].index, int(len(filt[~filt].index) * 0.99), replace=False)
                ] = True
            sampled_data = sampled_data[filt]

    sampledX = pf2(sampled_data, "Condition", rank)

    plotLabelsUMAP(sampledX, "cell_type", ax[0])

    return f
