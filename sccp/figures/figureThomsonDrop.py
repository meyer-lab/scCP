"""
Describes the model's reaction to the removal of specific pieces of the
dataset. It can clearly be observed for all cell types that had an identifying component that the
corresponding weight for the condition removed significantly dropped.
"""

from .common import getSetup
from ..imports import import_thomson
from ..factorization import pf2
from ..imports import import_thomson
from .commonFuncs.plotFactors import reorder_table
import pandas as pd
import numpy as np
import seaborn as sns
from ..gating import marker_genes_1, marker_genes_2
from anndata import AnnData
from matplotlib.axes import Axes
from scipy.stats import linregress


def makeFigure():
    rank = 20
    data = import_thomson()
    
    ax, f = getSetup((10, 10 * 9), (1 * 9, 2))
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
    DCCellGeneSet = marker_genes_1["Dendritic cells"]
    pDCGeneSet = marker_genes_2["pDCs"]
    TCellGeneSet = np.unique(marker_genes_1["T cells"] + marker_genes_2["T Cells"])
    CytoTCellGeneSet = marker_genes_2["Cytotoxic T"]
    memorytcellgeneset = marker_genes_2["Memory T"]
    cDCgeneset = marker_genes_2["cDCs"]
   
    plotDifferentialExpression(data, "CTRL4", "B Cells", bCellGeneSet, rank, *ax[0:2])
    plotDifferentialExpression(
        data,
        "CTRL4",
        "Naive B Cells",
        NaiveBCellGeneSet,
        rank,
        *ax[2:4],
        ct2=True,
    )
    plotDifferentialExpression(
        data, "CTRL4", "NK Cells", NKCellGeneSet, rank, *ax[4:6]
    )
    plotDifferentialExpression(data, "CTRL4", "DCs", DCCellGeneSet, rank, *ax[6:8])
    plotDifferentialExpression(
        data,
        "CTRL4",
        "pDCs",
        pDCGeneSet,
        rank,
        *ax[8:10],
        ct2=True,
    )
    plotDifferentialExpression(data, "CTRL4", "T Cells", TCellGeneSet, rank, *ax[10:12])
    plotDifferentialExpression(
        data, "CTRL4", "Cytotoxic T Cells", CytoTCellGeneSet, rank, *ax[12:14], ct2=True
    )
    plotDifferentialExpression(
        data, "CTRL4", "Memory T Cells", memorytcellgeneset, rank, *ax[14:16], ct2=True
    )
    plotDifferentialExpression(
        data, "CTRL4", "cDCs", cDCgeneset, rank, *ax[16:18], ct2=True
    )
    return f


def plotDifferentialExpression(
    data: AnnData,
    condition: str,
    cell_type: str,
    geneset: list[str],
    rank: int,
    *args: list[Axes],
    ct2:bool = False,
    override: tuple[int, int] = None,
    percent: int = 1,
):
    """
    Plots the differences in model weights of a gene set in the original data and the data with a certain percentage of the cell type removed. The determining component is the component that has the highest correlation with the number of cells in the cell type selected.
    Args:
        data: The AnnData object to be used
        condition: The condition to drop the cell type from
        cell_type: The cell type to be drop
        geneset: The marker genes
        rank: The rank to be used
        *args: The axes to be used
        ct2: Whether to use the second cell type
        override: The component to be used if not using the highest absolute value weight
        percent: The percentage of the cell type to be removed
    Returns:
        None
    """
    sampled_data = None
    ctarg = "Cell Type"
    if ct2:
        ctarg = "Cell Type2"
    idx = (data.obs[ctarg] != cell_type) | (data.obs["Condition"] != condition)
    false_idx = idx.index[idx == False]
    idx[
        np.random.choice(
            false_idx, size=int(len(false_idx) * (1 - percent)), replace=False
        )
    ] = True
    sampled_data = data[idx]

    origX = pf2(data, rank, doEmbedding=False)

    sampledX = pf2(sampled_data, rank, doEmbedding=False)
    most_exp_cmp = -1
    most_exp_cmp2 = -1
    yt = pd.Series(np.unique(origX.obs["Condition"]))
    yt2 = pd.Series(np.unique(sampledX.obs["Condition"]))

    numberOfCellType = []
    for txt in yt:
        numberOfCellType.append(
            len(data[(data.obs["Condition"] == txt) & (data.obs[ctarg] == cell_type)])
        )

    if not override: # Sum over the values of each component with the isolated gene factors and find the most expressed one
        # most_exp_cmp = np.argmax(np.sum(np.abs(isolated_orig_vals), axis=0))
        # most_exp_cmp2 = np.argmax(np.sum(np.abs(isolated_vals), axis=0))
        X = np.array(origX.uns["Pf2_A"])
        X2 = np.array(sampledX.uns["Pf2_A"])
        all_r2 = []
        all_r2_2 = []
        for i in range(X.shape[1]):
            _, _, r_value, _, _ = linregress(X[:, i], numberOfCellType)
            _, _, r_value2, _, _ = linregress(X2[:, i], numberOfCellType)
            all_r2.append(r_value ** 2)
            all_r2_2.append(r_value2 ** 2)
        most_exp_cmp = np.argmax(all_r2)
        most_exp_cmp2 = np.argmax(all_r2_2)
        print(f"All r2 original: {all_r2}")
        print(f"All r2 sampled: {all_r2_2}")
    else:
        most_exp_cmp = override[0]
        most_exp_cmp2 = override[1]

    X = np.array(origX.uns["Pf2_A"])
    X = X[:, most_exp_cmp]
    Y = np.array(sampledX.uns["Pf2_A"])
    Y = Y[:, most_exp_cmp2]

    assert yt.equals(yt2)

    print(f"Number of {cell_type}: {len(data[data.obs[ctarg] == cell_type])}")
    print(
        f"Number of {cell_type} in {condition}: {len(data[(data.obs['Condition'] == condition) & (data.obs[ctarg] == cell_type)])}"
    )

    args[0].scatter(X, Y)
    for i, txt in enumerate(yt):
        if txt == condition:
            args[0].annotate(txt, (X[i], Y[i]), fontsize=8)

    args[0].set_xlabel("Original Full Data")
    if percent < 1:
        args[0].set_ylabel(f"Sampled Data With {percent*100} Percent Of {cell_type} Removed From {condition}")
    else:
        args[0].set_ylabel(f"Sampled Data With {cell_type} Removed From {condition}")
    args[0].set_ylim(bottom=0)
    args[0].set_xlim(left=0)

    args[1].set_xlabel(f"Component {most_exp_cmp + 1} in Original Data Weight")
    args[1].set_ylabel(f"Number of {cell_type} per condition in Original Data")
    args[1].scatter(X, numberOfCellType)
    args[1].set_ylim(bottom=0)
    args[1].set_xlim(left=0)


def plotGeneFactorsIsolated(data: AnnData, ax: Axes, geneset: list[str], trim: bool = True):
    """
    Plots the geneset isolated from the Pf2 gene factors
    Args:
        data: The AnnData object to be used
        ax: The axes to plot to
        geneset: The marker genes
        trim: Whether to trim the data
    Returns:
        np.array: The isolated gene factors
    """
    rank = data.varm["Pf2_C"].shape[1]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    X = np.array(data.varm["Pf2_C"])
    yt = data.var.index.values

    if trim is True:
        max_weight = np.max(np.abs(X), axis=1)
        kept_idxs = max_weight > 0.08
        X = X[kept_idxs]
        yt = yt[kept_idxs]

    ind = reorder_table(X)
    ind = [ii for ii in ind if yt[ii] in geneset]
    yt = [yt[ii] for ii in ind]
    X = X / np.max(np.abs(X))
    X = X[ind]
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
