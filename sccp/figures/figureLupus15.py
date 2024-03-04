"""
Lupus: Plots of amount of cells and cell type distribution across all experiments
"""
from anndata import read_h5ad
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import population_bar_chart
from .figureLupus18 import getCellCountDF

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 10), (5, 5))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")

    df = X.obs[["Cell Type", "SLE_status", "Condition"]].reset_index(drop=True)

    dfCond = (
        df.groupby(["Condition"], observed=True).size().reset_index(name="Cell Number")
    )
    sns.histplot(data=dfCond, x="Cell Number", bins=15, color="k", ax=ax[0])
    ax[0].set(ylabel="# of Experiments")
    
    
    dfCellType = getCellCountDF(X, celltype="Cell Type", cellPerc=False)
    sns.boxplot(data=dfCellType, x="Cell Type", y="Count", hue="SLE_status", ax=ax[1])
    ax[1].set_xticklabels(labels=ax[1].get_xticklabels(), rotation=90)
    

    dfCellType = getCellCountDF(X, celltype="Cell Type2", cellPerc=False)
    sns.boxplot(data=dfCellType, x="Cell Type", y="Count", hue="SLE_status", ax=ax[2])
    ax[2].set_xticklabels(labels=ax[2].get_xticklabels(), rotation=90)
    
            
    dfCellType = getCellCountDF(X, celltype="Cell Type", cellPerc=True)
    sns.boxplot(data=dfCellType, x="Cell Type", y="Cell Type Percentage", hue="SLE_status", ax=ax[3])
    ax[3].set_xticklabels(labels=ax[3].get_xticklabels(), rotation=90)
    
    
    dfCellType = getCellCountDF(X, celltype="Cell Type2", cellPerc=True)
    sns.boxplot(data=dfCellType, x="Cell Type", y="Cell Type Percentage", hue="SLE_status", ax=ax[4])
    ax[4].set_xticklabels(labels=ax[4].get_xticklabels(), rotation=90)
    
    
    population_bar_chart(X, "Cell Type", "SLE_status", ax[5])
    
    
    for i, celltype in enumerate(np.unique(dfCellType["Cell Type"])):
        sns.histplot(
            data=dfCellType.loc[dfCellType["Cell Type"] == celltype],
            x="Cell Type Percentage",
            bins=15,
            color="k",
            ax=ax[i + 6],
        )
        ax[i + 6].set(title=celltype, ylabel="# of Experiments")

 

    return f


