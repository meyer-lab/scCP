"""
Lupus: Plots of amount of cells and cell type distribution across all experiments
"""
from anndata import read_h5ad
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import population_bar_chart
from .figureLupus17 import getCellCountPercDF


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 6), (4, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")
    
    celltype = ["Cell Type", "Cell Type2", "leiden"]
    cellperc = [True, False]
    label = ["Cell Type Percentage", "Cell Count"]
    plot = 0

    for i in range(len(celltype)):
        for j in range(len(cellperc)):
            cellPercDF = getCellCountPercDF(X, celltype=celltype[i], cellPerc=cellperc[j])
            sns.boxplot(data=cellPercDF, x="Cell Type", y=label[j], 
                        hue="Status", showfliers=False, ax=ax[plot])
            rotate_axis(ax[plot])
            plot+=1
       
    plotOverallCellCount(X, ax[6])
    f.delaxes(ax[7])   
            


    return f
    
def plotOverallCellCount(X, ax):
    df = X.obs[["SLE_status", "Condition"]].reset_index(drop=True)
    dfCond = df.groupby(["Condition","SLE_status"], observed=True).size().reset_index(name="Cell Count")
    
    sns.boxplot(data=dfCond, x="SLE_status", y="Cell Count", hue="SLE_status", showfliers=False, ax=ax)
    
    return 

def rotate_axis(ax):
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)