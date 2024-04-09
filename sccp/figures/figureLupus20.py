"""
Lupus
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

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((5, 5), (2, 2))


    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    cellPercDF = getCellCountPercDF(X, celltype="leiden", cellPerc=True)
    celltype = np.unique(cellPercDF["Cell Type"])
    sns.boxplot(data=cellPercDF, x="Cell Type", y="Cell Type Percentage", hue="Status", order=celltype, showfliers=False, ax=ax[0])
    ax[0].set_xticks(ax[0].get_xticks())
    ax[0].set_xticklabels(labels=ax[0].get_xticklabels(), rotation=90)

    cmp=28
    idx = len(np.unique(cellPercDF["Cell Type"]))
    cellCountDF = getCellCountPercDF(X, celltype="leiden", cellPerc=False)
    plotCmpPerCellCount(X, cmp, cellCountDF, ax[1:idx+2], cellPerc=False)


    return f


def getCellCountPercDF(X, celltype="Cell Type", cellPerc=True):
    """Determine cell count or cell type percentage per condition and patient"""
    df = X.obs[["Cell Type", "SLE_status", "Condition", "Cell Type2", "leiden"]].reset_index(drop=True)
    dfCond = (
        df.groupby(["CellType2", "Condition"], observed=True).size().reset_index(name="Cell Number")
    )
    dfCellType = (
        df.groupby([celltype, "Condition", "SLE_status"], observed=True)
        .size()
        .reset_index(name="Cell Count")
    )
    dfCellType["Cell Count"] = dfCellType["Cell Count"].astype("float")

    if cellPerc is True:
        for i, cond in enumerate(np.unique(df["Condition"])):
            dfCellType.loc[dfCellType["Condition"] == cond, dfCellType["Condition"] == cond,"Cell Count"] = (
                100
                * dfCellType.loc[dfCellType["Condition"] == cond, "Cell Count"].to_numpy()
                / dfCond.loc[dfCond["Condition"] == cond]["Cell Number"].to_numpy()
            )
        dfCellType.rename(columns={"Cell Count": "Cell Type Percentage"}, inplace=True)

    dfCellType.rename(columns={celltype: "Cell Type", "SLE_status": "Status"}, inplace=True)

    return dfCellType



    
# """
# Lupus
# """
# from anndata import read_h5ad
# from .common import (
#     subplotLabel,
#     getSetup,
# )
# import numpy as np
# import seaborn as sns
# import pandas as pd
# from scipy.stats import linregress, pearsonr, spearmanr

# def makeFigure():
#     """Get a list of the axis objects and create a figure."""
#     # Get list of axis objects
#     ax, f = getSetup((5, 5), (2, 2))


#     # Add subplot labels
#     subplotLabel(ax)

#     X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

#     cellPercDF = getCellCountPercDF(X, celltype="CM")
#     sns.boxplot(data=cellPercDF, x="Leiden Cluster", y="Cell Percentage", hue="Status", showfliers=False, ax=ax[0])
#     ax[0].set(title="CM")
    
#     cellPercDF = getCellCountPercDF(X, celltype="T4 EM")
#     sns.boxplot(data=cellPercDF, x="Leiden Cluster", y="Cell Percentage", hue="Status",  showfliers=False, ax=ax[1])
#     ax[1].set(title="T4 EM")
#     # ax[0].set_xticks(ax[0].get_xticks())
#     # ax[0].set_xticklabels(labels=ax[0].get_xticklabels(), rotation=90)



#     return f


# def getCellCountPercDF(X, celltype, cellPerc=True):
#     """Determine cell count or cell type percentage per condition and patient"""
#     df = X.obs[["Cell Type", "SLE_status", "Condition", "Cell Type2", "leiden"]].reset_index(drop=True)
    

#     df = df.loc[df["Cell Type2"] == celltype]
    
    
    
   
#     dfCond = (
#         df.groupby(["leiden","Condition", "SLE_status"], observed=True).size().reset_index(name="Cell Percentage")
#     )
    

#     print(dfCond)
    
#     total = np.sum(dfCond["Cell Percentage"])
    
#     dfCond["Cell Percentage"] = dfCond["Cell Percentage"]/total
#     print(dfCond)
  
#     leiden = dfCond["leiden"].to_numpy()
#     dfCond = dfCond.drop(columns=["leiden"])
#     dfCond["Leiden Cluster"] = leiden

    
    
#     # dfCellType = dfCellType.loc[dfCellType["leiden"] == "44"]
#     # print(dfCellType)
#     # a

#     # if cellPerc is True:
#     #     for i, cond in enumerate(np.unique(df["Condition"])):
#     #         dfCellType.loc[dfCellType["Condition"] == cond, "Cell Count"] = (
#     #             100
#     #             * dfCellType.loc[dfCellType["Condition"] == cond, "Cell Count"].to_numpy()
#     #             / dfCond.loc[dfCond["Condition"] == cond]["Cell Number"].to_numpy()
#     #         )
#     #     dfCellType.rename(columns={"Cell Count": "Cell Type Percentage"}, inplace=True)

#     dfCond.rename(columns={"SLE_status": "Status"}, inplace=True)

#     return dfCond

    