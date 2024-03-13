"""
Describes the model's reaction to the removal of specific pieces of the
dataset. It can clearly be observed for all cell types that had an identifying component that the
corresponding weight for the condition removed significantly dropped.
"""

from .common import getSetup
from ..imports import import_thomson
from ..factorization import pf2
from ..imports import import_thomson
import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from scipy.stats import linregress


def makeFigure():
    rank = 20
    data = import_thomson()
    
    ax, f = getSetup((6, 3), (1, 2))
   
    plotDifferentialExpression(data, "CTRL4", "B Cells", rank, *ax[0:2])
    # plotDifferentialExpression(
    #     data,
    #     "CTRL4",
    #     "Naive B Cells",
    #     rank,
    #     *ax[2:4],
    #     ct2=True,
    # )
    # plotDifferentialExpression(
    #     data, "CTRL4", "NK Cells", rank, *ax[4:6]
    # )
    # plotDifferentialExpression(data, "CTRL4", "DCs", rank, *ax[6:8])
    # plotDifferentialExpression(
    #     data,
    #     "CTRL4",
    #     "pDCs",
    #     rank,
    #     *ax[8:10],
    #     ct2=True,
    # )
    # plotDifferentialExpression(data, "CTRL4", "T Cells", rank, *ax[10:12])
    # plotDifferentialExpression(
    #     data, "CTRL4", "Cytotoxic T Cells", rank, *ax[12:14], ct2=True
    # )
    # plotDifferentialExpression(
    #     data, "CTRL4", "Memory T Cells", rank, *ax[14:16], ct2=True
    # )
    # plotDifferentialExpression(
    #     data, "CTRL4", "cDCs", rank, *ax[16:18], ct2=True, override=(14, 15)
    # )
    return f


def plotDifferentialExpression(
    data: AnnData,
    condition: str,
    cell_type: str,
    rank: int,
    *args: list[Axes],
    ct2:bool = False,
    override: tuple[int, int] = None,
):
    """
    Plots the differences in model weights of a gene set in the original data and the data. The determining component is the component that has the highest correlation with the number of cells in the cell type selected.
    Args:
        data: The AnnData object to be used
        condition: The condition to drop the cell type from
        cell_type: The cell type to be drop
        geneset: The marker genes
        rank: The rank to be used
        *args: The axes to be used
        ct2: Whether to use the second cell type
        override: The component to be used if not using the highest absolute value weight
    Returns:
        None
    """
    sampled_data = None
    ctarg = "Cell Type"
    if ct2:
        ctarg = "Cell Type2"
    idx = (data.obs[ctarg] != cell_type) | (data.obs["Condition"] != condition)
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

    all_r2 = []
    all_r2_2 = []
    if not override: # Use r^2 values to find the most important component
        X = np.array(origX.uns["Pf2_A"])
        X2 = np.array(sampledX.uns["Pf2_A"])
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

        all_r2 = [0] * rank
        X = np.array(origX.uns["Pf2_A"])
        _, _, r_value, _, _ = linregress(X[:, most_exp_cmp], numberOfCellType)
        all_r2[most_exp_cmp] = r_value ** 2

    X = np.array(origX.uns["Pf2_A"])
    X = X[:, most_exp_cmp]
    Y = np.array(sampledX.uns["Pf2_A"])
    Y = Y[:, most_exp_cmp2]

    assert yt.equals(yt2)

    print(f"Number of {cell_type}: {len(data[data.obs[ctarg] == cell_type])}")
    print(
        f"Number of {cell_type} in {condition}: {len(data[(data.obs['Condition'] == condition) & (data.obs[ctarg] == cell_type)])}"
    )

    colors = ['b'] * len(yt)
    for i, txt in enumerate(yt):
        if txt == condition:
            colors[i] = 'r'

    args[0].scatter(X, Y, c=colors)
    a, b = np.polyfit(X, Y, 1)
    args[0].axline(
        (0, b),
        slope=a,
        linestyle='--'
    )
    args[0].scatter([], [], c='b', label='Other Conditions')
    args[0].scatter([], [], c='r', label=condition)
    args[0].set_xlabel(f"Component {most_exp_cmp + 1} Weight (Full Data)")
    args[0].set_ylabel(f"Component {most_exp_cmp + 1} Weight (Sampled Data)")
    args[0].set_title(f"Full Data vs {cell_type} removed from {condition}")
    args[0].set_ylim(bottom=0)
    args[0].set_xlim(left=0)
    args[0].legend(loc="upper left")

    args[1].set_xlabel(f"Component {most_exp_cmp + 1} Weight (Full Data)")
    args[1].set_ylabel(f"Number of {cell_type} per condition (Full Data)")
    args[1].scatter(X, numberOfCellType)
    a, b = np.polyfit(X, numberOfCellType, 1)
    args[1].axline(
        (0, b),
        slope=a,
        label=f"R^2 value: {all_r2[most_exp_cmp]}",
        linestyle='--'
    )
    args[1].set_ylim(bottom=0)
    args[1].set_xlim(left=0)
    args[1].legend(loc="upper left")
