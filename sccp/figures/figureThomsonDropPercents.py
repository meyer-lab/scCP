"""
Plots the differences in the identifying component weight with different percentages of a cell type removed from the data. The determining component is the component that has the highest correlation with the number of cells in the cell type selected.
"""

from .common import getSetup
from ..imports import import_thomson
from ..imports import import_thomson
import numpy as np
from ..factorization import pf2
import pandas as pd


def makeFigure():
    rank = 20
    data = import_thomson()
    ax, f = getSetup((3, 3), (1, 1))
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

    plot_weights_across_percents(
        data, "B Cells", "CTRL4", 0, 1, 0.1, rank, bCellGeneSet, ax[0]
    )

    return f


def plot_weights_across_percents(
    data,
    cell_type,
    condition,
    percent_min,
    percent_max,
    percent_step,
    rank,
    geneset,
    ax,
):
    """
    Plots the raw weight of the identifying component across the percentages dropped
    """
    vals = {}
    for percent in np.arange(percent_min, percent_max, percent_step):
        print(f"Percent: {percent}")
        idx = (data.obs["Cell Type"] != cell_type) | (
            data.obs["Condition"] != condition
        )
        false_idx = idx.index[idx == False]
        idx[
            np.random.choice(
                false_idx, size=int(len(false_idx) * (1 - percent)), replace=False
            )
        ] = True
        sampled_data = data[idx]

        sampledX = pf2(sampled_data, rank, doEmbedding=False)
        gene_values = np.array(sampledX.varm["Pf2_C"])
        yt = data.var.index.values
        inds = []
        for i, gene in enumerate(yt):
            if gene in geneset:
                inds.append(i)
        gene_values = gene_values[inds]
        most_exp_cmp = np.argmax(np.sum(np.abs(gene_values), axis=0))
        Y = np.array(sampledX.uns["Pf2_A"])
        Y = Y[:, most_exp_cmp]
        labels = pd.Series(np.unique(sampledX.obs["Condition"]))
        for i, txt in enumerate(labels):
            if txt == condition:
                vals[percent] = Y[i]
                break

    percents = []
    weights = []
    for percent, weight in vals.items():
        percents.append(percent)
        weights.append(weight)
    weights = [x for _, x in sorted(zip(percents, weights))]
    percents = sorted(percents)
    ax.plot(percents, weights)
    min_val, max_val = min(weights), max(weights)
    padding = (max_val - min_val) * 0.1
    ax.set_ylim(min_val - padding, max_val + padding)
    ax.set_xlabel("Percent of B Cells Dropped")
    ax.set_ylabel(f"Weight of Identifying Component (Component {most_exp_cmp})")
