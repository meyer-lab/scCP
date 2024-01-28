from .common import getSetup
from ..imports import import_thomson
from ..factorization import pf2
from ..imports import import_thomson
from .commonFuncs.plotFactors import reorder_table
import pandas as pd
import numpy as np
import seaborn as sns

cmap = sns.diverging_palette(240, 10, as_cmap=True)


def makeFigure():
    rank = 20
    data = import_thomson()

    sampled_data = data[
        (data.obs["Cell Type"] != "B Cells") | (data.obs["Condition"] != "CTRL4")
    ]

    ax, f = getSetup((20, 10), (1, 2))
    bCellGeneSet = [
        "PXK",
        "MS4A1",
        "CD19",
        "CD74",
        "CD79A",
        "CD79B",
        "BANK1",
        "PTPRC",
        "CR2",
        "VPREB3",
    ]

    origX = pf2(data, rank, doEmbedding=False)

    sampledX = pf2(sampled_data, rank, doEmbedding=False)

    plotGeneFactorsIsolated(sampledX, ax[0], bCellGeneSet)

    X = np.array(origX.uns["Pf2_A"])
    X = X[:, 9]
    Y = np.array(sampledX.uns["Pf2_A"])
    Y = Y[:, 9]

    yt = pd.Series(np.unique(origX.obs["Condition"]))
    yt2 = pd.Series(np.unique(sampledX.obs["Condition"]))

    assert yt.equals(yt2)

    ax[1].scatter(X, Y, s=1)
    for i, txt in enumerate(yt):
        ax[1].annotate(txt, (X[i], Y[i]), fontsize=8)

    ax[1].set_xlabel("Original Full Data")
    ax[1].set_ylabel("Sampled Data With B Cells Removed From CTRL4")

    return f


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
