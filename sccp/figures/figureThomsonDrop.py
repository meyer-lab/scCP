from sccp.figures.common import (subplotLabel, 
getSetup, openPf2, savePf2, flattenData)
from sccp.imports.scRNA import ThompsonXA_SCGenesAD, tensorFy
from sccp.imports.gating import gateThomsonCells
from parafac2 import parafac2_nd
import umap
import numpy as np
import json
import seaborn as sns
import anndata as ad
from sccp.figures.commonFuncs.plotFactors import (
    plotFactors,
)
from sccp.figures.commonFuncs.plotUMAP import (
    plotCmpPerCellType,
    plotCmpUMAP,
    points
)
import pacmap


def makeFigure():
    data = ThompsonXA_SCGenesAD()
    rank = 30
    data.obs["cell_type"] = gateThomsonCells()

    ax, f = getSetup((10,7), (1, 1))
    subplotLabel(ax)

    sampled_data = data.copy()
    for drug in data.obs.Drugs.unique():
        for cell_type in data.obs.cell_type.unique():
            filt = (sampled_data.obs.cell_type != cell_type) | (
                    sampled_data.obs.Drugs != drug
                )
            filt[
                    np.random.choice(filt[~filt].index, int(len(filt[~filt].index) * 0.99), replace=False)
                ] = True
            sampled_data = sampled_data[filt]

    dataDF = tensorFy(sampled_data, "Drugs")
    weight, factors, projs, r2x = parafac2_nd(
        dataDF,
        rank=rank,
        random_state=1,
        )

    dataDF_flat = flattenData(dataDF)
    dataDF_flat['Cell Type'] = sampled_data.obs['cell_type'].values
    new_projs = np.concatenate(projs, axis=0)
    pf2Points = pacmap.PaCMAP().fit_transform(new_projs)

    points(pf2Points, labels=dataDF_flat["Cell Type"].values, ax=ax[0])
    ax[0].set(ylabel="UMAP2", xlabel="UMAP1")

    return f
