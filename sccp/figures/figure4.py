"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
import numpy as np
from .common import subplotLabel, getSetup, plotFactors, plotProj, plotR2X, plotCV, plotCondFactorsReorder
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import seaborn as sns
import pandas as pd

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 25), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    # data = ThompsonXA_SCGenes()
    rank = 3
    
    # weight = np.transpose([1, 2])
    # rank=2
        
    # weight, factors, projs, _ = parafac2_nd(
    #     data,
    #     rank=rank,
    #     random_state=1,
    # )

    a = [np.random.rand(rank)]
    print(np.shape(a))
    

    df = pd.DataFrame(data=np.transpose([np.random.rand(rank)]), columns=["Value"])
    df["Component"] = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    ax.tick_params(axis="y", rotation=0)
    print(df)
    
    
    
    

    # print(weight)
    # print(np.shape(weight))
    # plotFactors(factors, data, ax[0:3], reorder=(0, 2), trim=(2,), saveGenes=False)
    
    # plotCondFactorsReorder(factors, data, ax[3])
    
    
    # df = pd.DataFrame(data=weight, columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)])
    sns.barplot(data=df,x="Component", y="Value", ax=ax[0])
    # pd.melt(df, )

    # cellcountDF = dataDF.groupby(["Condition"]).size().reset_index(name="Cell Count") 
    # sns.barplot(data=cellcountDF, x="Condition", y="Cell Count", ax=ax)
    # ax.tick_params(axis="x", rotation=90)
    


    # plotCV(data, rank+3, trainPerc=0.75, ax=ax[2])
    # plotR2X(data, rank+3, ax=ax[3])

    return f
