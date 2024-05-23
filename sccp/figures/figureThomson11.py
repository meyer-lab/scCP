"""
Thomson
"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr
from .commonFuncs.plotGeneral import cell_count_perc_df, rotate_xaxis
from matplotlib.axes import Axes
import anndata


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 9), (5, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad", backed="r")

    celltype_count_perc_df = cell_count_perc_df(X, celltype="Cell Type2")

    for i in range(20):
        plot_correlation_cmp_cell_count_perc(
            X, i+1, celltype_count_perc_df, ax[i], cellPerc=False
        )

    return f


def plot_correlation_cmp_cell_count_perc(
    X: anndata, cmp: int, cellcountDF: pd.DataFrame, ax: Axes, cellPerc=True
):
    """Plot component weights by cell type count or percentage for a cell type"""
    yt = np.unique(X.obs["Condition"])
    factorsA = np.array(X.uns["Pf2_A"])
    factorsA = factorsA[:, cmp - 1]
    if cellPerc is True:
        cellPerc = "Cell Type Percentage"
    else:
        cellPerc = "Cell Count"
    totaldf = pd.DataFrame([])
    correlationdf = pd.DataFrame([])
    cellcountDF["Condition"] = pd.Categorical(cellcountDF["Condition"], yt)
    for i, celltype in enumerate(np.unique(cellcountDF["Cell Type"])):
        for j, cond in enumerate(np.unique(cellcountDF["Condition"])):
            smalldf = cellcountDF.loc[
                (cellcountDF["Condition"] == cond)
                & (cellcountDF["Cell Type"] == celltype)
            ]
            
            if smalldf.empty is False:
                smalldf = smalldf.assign(Cmp=factorsA[j])
            else:
                smalldf = pd.DataFrame(
                    {
                        "Condition": cond,
                        "Cell Type": celltype,
                        cellPerc: [0],
                        "Cmp": [factorsA[j]],
                    }
                )

            totaldf = pd.concat([totaldf, smalldf])
            
            
        df = totaldf.loc[totaldf["Cell Type"] == celltype]
        _, _, r_value, _, _ = linregress(df["Cmp"], df[cellPerc])
        
        pearson = pearsonr(df["Cmp"], df[cellPerc])[0]
        spearman = spearmanr(df["Cmp"], df[cellPerc])[0]
        
        correl = [np.round(r_value**2, 3), spearman, pearson]
        test = ["Spearman"]

        for k in range(1):
            correlationdf = pd.concat(
                [
                    correlationdf,
                    pd.DataFrame(
                        {
                            "Cell Type": celltype,
                            "Correlation": [test[k]],
                            "Correlation": [correl[k]],
                        }
                    ),
                ]
            )
    sns.barplot(
        data=correlationdf, x="Cell Type", y="Correlation", ax=ax
    )
    rotate_xaxis(ax)
    ax.set(title=f"Cmp. {cmp} V. {cellPerc}")