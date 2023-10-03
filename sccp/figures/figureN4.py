from sccp.figures.common import (
    subplotLabel,
    getSetup,
    plotFactorsRelaxed,
    plotFactors,
    plotR2X,
    plotCV,
    plotWeight,
    openPf2,
    savePf2,
    plotCellTypeUMAP,
    flattenData,
    openUMAP,
)
from sccp.imports.scRNA import ThompsonXA_SCGenesAD, tensorFy
from sccp.imports.gating import gateThomsonCells
from parafac2 import parafac2_nd
import umap
import numpy as np
import json
import seaborn as sns


def makeFigure():
    ax, f = getSetup((10, 7), (1, 1))
    subplotLabel(ax)
    data = ThompsonXA_SCGenesAD()
    rank = 30
    data.obs["cell_type"] = gateThomsonCells(rank)
    sampled_data = data.copy()
    for drug in data.obs.Drugs.unique():
        for cell_type in data.obs.cell_type.unique():
            filt = (sampled_data.obs.cell_type != cell_type) | (
                sampled_data.obs.Drugs != drug
            )
            filt.loc[
                np.random.choice(filt[~filt].index, int(len(filt[~filt].index) * 0.99))
            ] = True
            sampled_data = sampled_data[filt]
    dataDF = tensorFy(sampled_data, "Drugs")
    weight, factors, projs, r2x = parafac2_nd(
        dataDF,
        rank=rank,
        random_state=1,
    )
    dataDF_flat = flattenData(dataDF)
    dataDF_flat["Cell Type"] = sampled_data.obs["cell_type"].values
    plotCellTypeUMAP(
        umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0)), dataDF_flat, ax[0]
    )
    return f
