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

    ax, f = getSetup((4, 24), (2, 1))

    origX = pf2(data, rank, doEmbedding=False)

    plotConditionsFactorsIsolated(
        origX, ax[0], groupDrugs(origX.obs["Condition"]), ThomsonNorm=True, isolatedCol=9
    )
    # plotCellState(origX, ax[1])
    # bCellGeneSet = ["BANK1", "CD79A", "CD79B", "MS4A1", "VPREB3"]
    # plotGeneFactorsIsolated(origX, ax[2], bCellGeneSet)

    sampledX = pf2(sampled_data, rank, doEmbedding=False)

    plotConditionsFactorsIsolated(
        sampledX, ax[1], groupDrugs(origX.obs["Condition"]), ThomsonNorm=True, isolatedCol=9
    )
    # plotCellState(sampledX, ax[4])
    # plotGeneFactorsIsolated(sampledX, ax[5], bCellGeneSet)

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
    print(X)
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


def plotConditionsFactorsIsolated(
    data: AnnData,
    ax: Axes,
    cond_group_labels: Optional[pd.Series] = None,
    ThomsonNorm=False,
    groupConditions=False,
    isolatedCol=None,
):
    """Plots Pf2 condition factors"""
    pd.set_option("display.max_rows", None)
    yt = pd.Series(np.unique(data.obs["Condition"]))
    X = np.array(data.uns["Pf2_A"])

    X = np.log10(X)
    if ThomsonNorm is True:
        controls = yt.str.contains("CTRL")
        XX = X[controls]
    else:
        XX = X

    X -= np.median(XX, axis=0)
    X /= np.std(XX, axis=0)

    ind = reorder_table(X)
    X = X[ind]
    yt = yt.iloc[ind]

    if cond_group_labels is not None:
        cond_group_labels = cond_group_labels.iloc[ind]
        if groupConditions is True:
            ind = cond_group_labels.argsort()
            cond_group_labels = cond_group_labels.iloc[ind]
            X = X[ind]
            yt = yt.iloc[ind]
        ax.tick_params(axis="y", which="major", pad=20, length=0)
        # extra padding to leave room for the row colors
        # get list of colors for each label:
        colors = sns.color_palette(
            n_colors=pd.Series(cond_group_labels).nunique()
        ).as_hex()
        lut = {}
        legend_elements = []
        for index, group in enumerate(pd.Series(cond_group_labels).unique()):
            lut[group] = colors[index]
            legend_elements.append(Patch(color=colors[index], label=group))
        row_colors = pd.Series(cond_group_labels).map(lut)
        for iii, color in enumerate(row_colors):
            ax.add_patch(
                plt.Rectangle(
                    xy=(-0.05, iii),
                    width=0.05,
                    height=1,
                    color=color,
                    lw=0,
                    transform=ax.get_yaxis_transform(),
                    clip_on=False,
                )
            )
        # add a little legend
        ax.legend(handles=legend_elements, bbox_to_anchor=(0.18, 1.07))

    xticks = [f"Cmp. {i}" for i in np.arange(1, X.shape[1] + 1)]
    if isolatedCol is None:
        sns.heatmap(
            data=X,
            xticklabels=xticks,
            yticklabels=yt,
            ax=ax,
            center=0,
            cmap=cmap,
        )
    else:
        X = X[:, isolatedCol]
        sns.heatmap(
            data=X[:, None],
            xticklabels=[isolatedCol + 1],
            yticklabels=yt,
            ax=ax,
            center=0,
            cmap=cmap,
        )
    ax.tick_params(axis="y", rotation=0)
