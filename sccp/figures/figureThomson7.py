"""
Investigation of raw data for Thomson dataset
"""
from .common import subplotLabel, getSetup
from ..imports.gating import gateThomsonCells
import seaborn as sns
import numpy as np

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 6), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]

    dataDF["Cell Type"] = gateThomsonCells()
    
    dfCond = dataDF.groupby(["Condition"]).size().reset_index(name="Cell Number") 
    sns.histplot(data=dfCond, x="Cell Number", bins=15, color="k", ax=ax[0])
    ax[0].set(ylabel="# of Experiments")
    
    dfCellType = dataDF.groupby(["Cell Type", "Condition"]).size().reset_index(name="Count") 
    for i, cond in enumerate(data.condition_labels):
        dfCellType.loc[dfCellType["Condition"] == cond, "Count"] = 100 * dfCellType.loc[dfCellType["Condition"] == cond, "Count"].to_numpy()/dfCond.loc[dfCond["Condition"] == cond]["Cell Number"].to_numpy()
    
    dfCellType.rename(columns={"Count": "Cell Type Percentage"}, inplace=True)
    for i, celltype in enumerate(np.unique(dfCellType["Cell Type"])):
        sns.histplot(data=dfCellType.loc[dfCellType["Cell Type"] == celltype], x="Cell Type Percentage", bins=15, color="k", ax=ax[i+1])
        ax[i+1].set(title=celltype, ylabel="# of Experiments")

    return f
