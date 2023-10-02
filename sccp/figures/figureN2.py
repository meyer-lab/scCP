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
    ax, f = getSetup((18, 16), (2, 2))
    subplotLabel(ax)

    data = ThompsonXA_SCGenesAD()
    rank = 30

    data.obs["cell_type"] = gateThomsonCells(rank)

    cell_type = data.obs.cell_type.unique()[1]
    drug = data.obs.Drugs.unique()[0]
    print(cell_type)
    print(len(data))
    print(drug)
    data_subset = data[(data.obs.cell_type != cell_type) | (data.obs.Drugs != drug)]
    print(len(data_subset))
    dataDF = tensorFy(data_subset, "Drugs")
    weight, factors, projs, r2x = parafac2_nd(
        dataDF,
        rank=rank,
        random_state=1,
    )

    plotFactors(factors, dataDF, ax[0:3], reorder=(0, 2), trim=(2,))
    dataDF_flat = flattenData(dataDF)
    dataDF_flat["Cell Type"] = data_subset.obs["cell_type"].values
    plotCellTypeUMAP(
        umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0)), dataDF_flat, ax[3]
    )

    f.suptitle(
        f"Pf2 results + UMAP when dropping {cell_type} from {drug} treatment",
        fontsize=35,
    )

    return f
