from .common import getSetup
from ..imports import import_thomson
from ..factorization import pf2
from ..imports import import_thomson
from .commonFuncs.plotFactors import reorder_table
import pandas as pd
import numpy as np
import seaborn as sns
from ..gating import marker_genes_1, marker_genes_2

cmap = sns.diverging_palette(240, 10, as_cmap=True)


def makeFigure():
    rank = 20
    data = import_thomson()

    print(data.obs["Cell Type2"].unique())

    ax, f = getSetup((20, 10 * 9), (1 * 9, 4))
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
    NaiveBCellGeneSet = marker_genes_2["B Naive"]
    NKCellGeneSet = np.unique(marker_genes_1["Natural Killers"] + marker_genes_2["NK"])
    DCCellGeneSet = np.unique(
        marker_genes_1["Dendritic cells"] + marker_genes_2["cDCs"]
    )
    pDCGeneSet = marker_genes_2["pDCs"]
    TCellGeneSet = np.unique(marker_genes_1["T cells"] + marker_genes_2["T Cells"])
    CytoTCellGeneSet = marker_genes_2["Cytotoxic T"]
    memorytcellgeneset = marker_genes_2["Memory T"]
    cDCgeneset = marker_genes_2["cDCs"]

    plotDifferentialExpression(data, "CTRL4", "B Cells", bCellGeneSet, rank, *ax[0:4])
    plotDifferentialExpression(
        data,
        "CTRL4",
        "Naive B Cells",
        NaiveBCellGeneSet,
        rank,
        *ax[4:8],
        ct2=True,
    )
    plotDifferentialExpression(data, "CTRL4", "NK Cells", NKCellGeneSet, rank, *ax[8:12])
    plotDifferentialExpression(
        data, "CTRL4", "DCs", DCCellGeneSet, rank, *ax[12:16]
    )
    plotDifferentialExpression(
        data, "CTRL4", "pDCs", pDCGeneSet, rank, *ax[16:20], ct2=True,
    )
    plotDifferentialExpression(data, "CTRL4", "T Cells", TCellGeneSet, rank, *ax[20:24])
    plotDifferentialExpression(
        data, "CTRL4", "Cytotoxic T Cells", CytoTCellGeneSet, rank, *ax[24:28], ct2=True
    )
    plotDifferentialExpression(
        data, "CTRL4", "Memory T Cells", memorytcellgeneset, rank, *ax[28:32], ct2=True
    )
    plotDifferentialExpression(
        data, "CTRL4", "cDCs", cDCgeneset, rank, *ax[32:36], ct2=True
    )
    return f


def plotDifferentialExpression(
    data,
    condition,
    cell_type,
    geneset,
    rank,
    *args,
    ct2=False,
    override=None
):
    sampled_data = None
    if not ct2:
        sampled_data = data[
            (data.obs["Cell Type"] != cell_type) | (data.obs["Condition"] != condition)
        ]
    else:
        sampled_data = data[
            (data.obs["Cell Type2"] != cell_type) | (data.obs["Condition"] != condition)
        ]

    origX = pf2(data, rank, doEmbedding=False)

    sampledX = pf2(sampled_data, rank, doEmbedding=False)

    isolated_orig_vals = plotGeneFactorsIsolated(origX, args[0], geneset)
    isolated_vals = plotGeneFactorsIsolated(sampledX, args[1], geneset)
    most_exp_cmp = -1
    most_exp = -1
    most_exp_cmp2 = -1
    most_exp2 = -1

    if override is None:
        for i in range(isolated_orig_vals.shape[1]):
            exp_sum = np.sum(np.abs(isolated_orig_vals[:, i]))
            if exp_sum > most_exp:
                most_exp = exp_sum
                most_exp_cmp = i
        for i in range(isolated_vals.shape[1]):
            exp_sum = np.sum(np.abs(isolated_vals[:, i]))
            if exp_sum > most_exp2:
                most_exp2 = exp_sum
                most_exp_cmp2 = i
    else:
        most_exp_cmp = override[0]
        most_exp_cmp2 = override[1]
    X = np.array(origX.uns["Pf2_A"])
    X = X[:, most_exp_cmp]
    Y = np.array(sampledX.uns["Pf2_A"])
    Y = Y[:, most_exp_cmp2]

    yt = pd.Series(np.unique(origX.obs["Condition"]))
    yt2 = pd.Series(np.unique(sampledX.obs["Condition"]))

    assert yt.equals(yt2)
    if not ct2:
        print(f"Number of {cell_type}: {len(data[data.obs['Cell Type'] == cell_type])}")
        print(f"Number of {cell_type} in {condition}: {len(data[(data.obs['Condition'] == condition) & (data.obs['Cell Type'] == cell_type)])}")
    else:
        print(f"Number of {cell_type}: {len(data[data.obs['Cell Type2'] == cell_type])}")
        print(f"Number of {cell_type} in {condition}: {len(data[(data.obs['Condition'] == condition) & (data.obs['Cell Type2'] == cell_type)])}")

    args[2].scatter(X, Y, s=1)
    numberOfCellType = []
    for i, txt in enumerate(yt):
        if not ct2:
            numberOfCellType.append(
                len(
                    data[
                        (data.obs["Condition"] == txt)
                        & (data.obs["Cell Type"] == cell_type)
                    ]
                )
            )
        else:
            numberOfCellType.append(
                len(
                    data[
                        (data.obs["Condition"] == txt)
                        & (data.obs["Cell Type2"] == cell_type)
                    ]
                )
            )
        args[2].annotate(txt, (X[i], Y[i]), fontsize=8)

    args[2].set_xlabel("Original Full Data")
    args[2].set_ylabel(f"Sampled Data With {cell_type} Removed From {condition}")

    args[3].set_xlabel(f"Component {most_exp_cmp + 1} in Original Data Weight")
    args[3].set_ylabel(f"Number of {cell_type} per condition in Original Data")
    args[3].scatter(X, numberOfCellType)
    args[3].set_ylim(bottom=0)
    args[3].set_xlim(left=0)


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
    return X
