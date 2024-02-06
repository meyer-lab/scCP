"""
Lupus: Plots of amount of cells and cell type distribution across all experiments
"""
from anndata import read_h5ad
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import population_bar_chart


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 15), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted.h5ad", backed="r")

    df = X.obs[["Cell Type", "SLE_status", "Condition"]].reset_index(drop=True)

    dfCond = (
        df.groupby(["Condition"], observed=True).size().reset_index(name="Cell Number")
    )
    _, count = np.unique(X.obs["Condition"], return_counts=True)
    Amatrix= X.uns["Pf2_A"]
    for i in range(Amatrix.shape[1]):
        Amatrix[:, i] /= count
        
    X.uns["Pf2_A"]= Amatrix
        
    dfCond = dfCond.sort_values(by=["Condition"])
    condNames = pd.Series(np.unique(X.obs["Condition"]))
    _, idx = np.unique(condNames, return_index=True)
    dfCond = dfCond.iloc[idx] 

    condFactors = np.array(X.uns["Pf2_A"])

    for i in range(12):
        ax[i].scatter(x=condFactors[:, i], y=dfCond["Cell Number"])
        ax[i].set(xlabel=f"Condition Factors Weight for Cmp. {i+1}", ylabel="Cell Number per Experiment")

  


    # dfCellType = (
    #     df.groupby(["Cell Type", "Condition"], observed=True)
    #     .size()
    #     .reset_index(name="Count")
    # )
    # dfCellType["Count"] = dfCellType["Count"].astype("float")
    # for i, cond in enumerate(pd.unique(df["Condition"])):
    #     dfCellType.loc[dfCellType["Condition"] == cond, "Count"] = (
    #         100
    #         * dfCellType.loc[dfCellType["Condition"] == cond, "Count"].to_numpy()
    #         / dfCond.loc[dfCond["Condition"] == cond]["Cell Number"].to_numpy()
    #     )

    # dfCellType.rename(columns={"Count": "Cell Type Percentage"}, inplace=True)
    # for i, celltype in enumerate(np.unique(dfCellType["Cell Type"])):
    #     sns.histplot(
    #         data=dfCellType.loc[dfCellType["Cell Type"] == celltype],
    #         x="Cell Type Percentage",
    #         bins=15,
    #         color="k",
    #         ax=ax[i + 1],
    #     )
    #     ax[i + 1].set(title=celltype, ylabel="# of Experiments")

    # population_bar_chart(X, "Cell Type", "SLE_status", ax[12])

    return f
