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
from scipy.stats import linregress

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 16), (5, 4))


    # Add subplot labels
    subplotLabel(ax)
    
    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    cellPercDF = getCellCountDF(X, celltype="Cell Type2", cellPerc=False)
    
    cmp=22
    idx = len(np.unique(cellPercDF["Cell Type"]))
    plotCmpPerCellCount(X, cmp, cellPercDF, ax[0:idx], cellPerc=False)
    
    return f


def getCellCountDF(X, celltype="Cell Type", cellPerc=True):
    
    df = X.obs[["Cell Type", "SLE_status", "Condition", "Cell Type2"]].reset_index(drop=True)

    dfCond = (
        df.groupby(["Condition"], observed=True).size().reset_index(name="Cell Number")
    )
    dfCellType = (
        df.groupby([celltype, "Condition", "SLE_status"], observed=True)
        .size()
        .reset_index(name="Count")
    )
    dfCellType["Count"] = dfCellType["Count"].astype("float")
    
    if cellPerc is True:
        for i, cond in enumerate(np.unique(df["Condition"])):
            dfCellType.loc[dfCellType["Condition"] == cond, "Count"] = (
                100
                * dfCellType.loc[dfCellType["Condition"] == cond, "Count"].to_numpy()
                / dfCond.loc[dfCond["Condition"] == cond]["Cell Number"].to_numpy()
            )
        dfCellType.rename(columns={"Count": "Cell Type Percentage"}, inplace=True)
        
    dfCellType.rename(columns={celltype: "Cell Type"}, inplace=True)
    
    return dfCellType


def plotCmpPerCellCount(X, cmp, cellcountDF, ax, cellPerc=True):
    yt = np.unique(X.obs["Condition"])
    factorsA = np.array(X.uns["Pf2_A"])
    factorsA = factorsA[:, cmp-1]
    
    if cellPerc is True:
        cellPerc = "Cell Type Percentage"
    else:
        cellPerc = "Count"
        
    totaldf = pd.DataFrame([])
    cellcountDF["Condition"] = pd.Categorical(cellcountDF["Condition"], yt)
    
    for i, celltype in enumerate(np.unique(cellcountDF["Cell Type"])):
        for j, cond in enumerate(np.unique(cellcountDF["Condition"])):
            status = np.unique(cellcountDF.loc[cellcountDF["Condition"] == cond]["SLE_status"])
            smalldf = cellcountDF.loc[(cellcountDF["Condition"] == cond) & (cellcountDF["Cell Type"] == celltype)]
        
            if smalldf.empty is False: 
                smalldf = smalldf.assign(Cmp=factorsA[j])
            else:
                smalldf = pd.DataFrame({"Condition": cond, "Cell Type": celltype, "SLE_status": status,
                                                  cellPerc: 0, "Cmp": factorsA[j]})
            
            totaldf = pd.concat([totaldf, smalldf])
            
        df = totaldf.loc[totaldf["Cell Type"] == celltype]   
        _, _, r_value, _, _ = linregress(df["Cmp"], df[cellPerc])
        
        sns.scatterplot(data=df, x="Cmp", y=cellPerc, hue="SLE_status", ax=ax[i])
        ax[i].set(title=f"{celltype}: R2 Value - {np.round(r_value**2, 3)}", xlabel=f"Cmp. {cmp}")