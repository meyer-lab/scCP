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
from .commonFuncs.plotGeneral import (
    plotGenePerCellType,
)
import numpy as np
import seaborn as sns
import scipy.cluster.hierarchy as sch

cmap = sns.diverging_palette(240, 10, as_cmap=True)


def makeFigure():
    rank = 20
    data = import_thomson()

    sampled_data = data[
        (data.obs["Cell Type"] != "B Cells") | (data.obs["Condition"] != "CTRL4")
    ]

    ax, f = getSetup((12, 12), (2, 3))

    origX = pf2(data, rank, doEmbedding=False)

    plotConditionsFactors(
        origX, ax[0], groupDrugs(origX.obs["Condition"]), ThomsonNorm=True
    )
    plotCellState(origX, ax[1])
    bCellGeneSet = ["BANK1", "CD79A", "CD79B", "MS4A1", "VPREB3"]
    plotGeneFactorsIsolated(origX, ax[2], bCellGeneSet)

    sampledX = pf2(sampled_data, rank, doEmbedding=False)

    plotConditionsFactors(
        sampledX, ax[3], groupDrugs(origX.obs["Condition"]), ThomsonNorm=True
    )
    plotCellState(sampledX, ax[4])
    plotGeneFactorsIsolated(sampledX, ax[5], bCellGeneSet)

    return f


def reorder_table(projs: np.ndarray) -> np.ndarray:
    """Reorder a table's rows using heirarchical clustering"""
    assert projs.ndim == 2
    Z = sch.linkage(projs, method="complete", metric="cosine", optimal_ordering=True)
    return sch.leaves_list(Z)


def plotGeneFactorsIsolated(data, ax, geneset, trim=True):
    """Plots Pf2 gene factors"""
    rank = data.varm["Pf2_C"].shape[1]
    X = np.array(data.varm["Pf2_C"])
    yt = data.var.index.values

    if trim is True:
        max_weight = np.max(np.abs(X), axis=1)
        kept_idxs = max_weight > 0.08
        X = X[kept_idxs]
        yt = yt[kept_idxs]

    ind = reorder_table(X)
    n_ind = []
    for ii in ind:
        if yt[ii] in geneset:
            n_ind.append(ii)
    ind = n_ind
    yt = [yt[ii] for ii in ind]
    X = X[ind]
    X = X / np.max(np.abs(X))
    xticks = [f"Cmp. {i}" for i in np.arange(1, rank)]

    sns.heatmap(
        data=X,
        xticklabels=xticks,
        yticklabels=yt,
        ax=ax,
        center=0,
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Components by Gene")
