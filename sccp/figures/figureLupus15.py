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
    ax, f = getSetup((5, 5), (2, 1))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")

    df = X.obs[["Cell Type", "SLE_status", "Condition", "Cell Type2"]].reset_index(drop=True)
    
    # a = df.loc[df["Cell Type2"] == "T4 Reg"]
    # print(a)
    # print(np.sum(a["Count"]))

    dfCond = (
        df.groupby(["Condition"], observed=True).size().reset_index(name="Cell Number")
    )
    # # sns.histplot(data=dfCond, x="Cell Number", bins=15, color="k", ax=ax[0])
    # # ax[0].set(ylabel="# of Experiments")

    dfCellType = (
        df.groupby(["Cell Type2", "Condition", "SLE_status"], observed=True)
        .size()
        .reset_index(name="Count")
    )
    print(dfCellType)
    dfCellType["Count"] = dfCellType["Count"].astype("float")
    
    
    for i, cond in enumerate(pd.unique(df["Condition"])):
        dfCellType.loc[dfCellType["Condition"] == cond, "Count"] = (
            100
            * dfCellType.loc[dfCellType["Condition"] == cond, "Count"].to_numpy()
            / dfCond.loc[dfCond["Condition"] == cond]["Cell Number"].to_numpy()
        )
        
    
    dfCellType.rename(columns={"Count": "Cell Type Percentage"}, inplace=True)
  

    
    sns.boxplot(data=dfCellType, x="Cell Type2", y="Cell Type Percentage", hue="SLE_status", ax=ax[0])
    ax[0].set_xticklabels(labels=ax[0].get_xticklabels(), rotation=90)

    
    
    
    
    # sns.boxplot(data=dfCellType, x="Cell Type", y="Count", hue="SLE_status", ax=ax[0],showfliers=False)
    
    # dfCellType = (
    #     df.groupby(["Cell Type2", "SLE_status", "Condition"], observed=True)
    #     .size()
    #     .reset_index(name="Count")
    # )
    # dfCellType["Count"] = dfCellType["Count"].astype("float")
    
    # sns.boxplot(data=dfCellType2, x="Cell Type2", y="Count", hue="SLE_status", ax=ax[1], showfliers=False)
    
    # ax[0].set_xticklabels(labels=ax[0].get_xticklabels(), rotation=90)
    # ax[1].set_xticklabels(labels=ax[1].get_xticklabels(), rotation=90)
    
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
