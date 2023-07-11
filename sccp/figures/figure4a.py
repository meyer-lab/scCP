"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
import numpy as np
from .common import subplotLabel, getSetup, plotFactors, plotProj, flattenData, plotR2X, plotCV
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import seaborn as sns


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 10), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    
    dataDF = flattenData(data)
    
    cellcountDF = dataDF.groupby(["Condition"]).size().reset_index(name="Cell Count") 
    
    print(cellcountDF)

    # for j, cond in enumerate(np.unique(dataDF["Condition"].values)):
    #     df = celltypeDF.loc[celltypeDF["Condition"] == cond] 
    #     perc = df["Count"].values / np.sum(df["Count"].values)
    #     celltypeDF.loc[celltypeDF["Condition"] == cond, "Count"] = perc
            
    sns.barplot(data=cellcountDF, x="Condition", y="Cell Count", ax=ax[0])
    ax[0].tick_params(axis="x", rotation=90)
    
    return f