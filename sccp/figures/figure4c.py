"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup, plotCmpUMAP, openPf2, flattenData, flattenWeightedProjs
from ..imports.scRNA import ThompsonXA_SCGenes
import umap 
import numpy as np
import os
from os.path import join
import umap.plot 
import seaborn as sns
import pandas as pd

path_here = os.path.dirname(os.path.dirname(__file__))



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((14, 14), (3, 3))

    # Add subplot labels
    subplotLabel(ax)
    
    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    rank = 30
    dataDF = flattenData(data)
    print(dataDF[(dataDF["Condition"] == "Dexrazoxane HCl (ICRF-187, ADR-529)")])
    dataDF["Cell Type"] = np.load(join(path_here, "data/Thomson/Thomson_CellTypes.npy"), allow_pickle=True)

    # UMAP dimension reduction
    weight, factors, projs = openPf2(rank, "Thomson")
    pf2Points = umap.UMAP(random_state=1).fit(projs)
        
    # umap.plot.points(pf2Points, labels=dataDF["Cell Type"].values, ax=ax[0])
    

    weightedProjDF = flattenWeightedProjs(data, factors, projs)
    weightedProjDF["Cell Type"] = dataDF["Cell Type"].values
    
    # df1 = dataDF[(dataDF["Condition"] == "Triamcinolone Acetonide")]
    # df2 = dataDF[(dataDF["Condition"] == "CTRL1")]
    
    # df = pd.concat([df1, df2])
    
    # drugs = [
    #     "Triamcinolone Acetonide",
    #     "Budesonide",
    #     "Loteprednol etabonate",
    #     "Betamethasone Valerate",
    #     "Alprostadil",
    #     "Meprednisone",
    #     "CTRL1",
    #     "CTRL2",
    #     "CTRL3",
    #     "CTRL4"
    # ]
    
    
    # for i, drugz in enumerate(drugs):
    #     df1 = dataDF[(dataDF["Condition"] == drugz)]
    #     sns.boxplot(data=df1[["MS4A1", "Cell Type", "Condition"]], x="MS4A1", y="Cell Type", hue="Condition", ax=ax[i])
    
    # sns.histplot(data=df1[["NKG7", "Cell Type"]], x="NKG7", hue="Cell Type", multiple="stack",  ax=ax[1])
    
    # sns.boxplot(data=weightedProjDF, x= "Cmp. 30", y= "Cell Type", ax=ax[1])
    
    # df1 = dataDF[(dataDF["Condition"] == "Triamcinolone Acetonide")]
    # df2 = dataDF[(dataDF["Condition"] == "CTRL1")]
    
    # df = pd.concat([df1, df2])
    
    # sns.boxplot(data=df[["CD163", "Cell Type", "Condition"]], x="CD163", y="Cell Type", hue="Condition", ax=ax[2])
    
    sns.violinplot(data=weightedProjDF, x= "Cmp. 5", y= "Cell Type", ax=ax[6])
    sns.boxplot(data=weightedProjDF, x= "Cmp. 5", y= "Cell Type", ax=ax[7])
    sns.barplot(data=weightedProjDF, x= "Cmp. 5", y= "Cell Type", ax=ax[8])
    
    df1 = dataDF[(dataDF["Condition"] == "Triamcinolone Acetonide")]
    df2 = dataDF[(dataDF["Condition"] == "CTRL1")]
    
    df = pd.concat([df1, df2])
    
    
    sns.boxplot(data=df[["GNLY", "Cell Type", "Condition"]], x="GNLY", y="Cell Type", hue="Condition", ax=ax[0])
    sns.boxplot(data=df[["NKG7", "Cell Type", "Condition"]], x="NKG7", y="Cell Type", hue="Condition", ax=ax[1])
    
    sns.barplot(data=df[["GNLY", "Cell Type", "Condition"]], x="GNLY", y="Cell Type", hue="Condition", ax=ax[2])
    sns.barplot(data=df[["NKG7", "Cell Type", "Condition"]], x="NKG7", y="Cell Type", hue="Condition", ax=ax[3])
    
    sns.violinplot(data=df[["GNLY", "Cell Type", "Condition"]], x="GNLY", y="Cell Type", hue="Condition",split=True, ax=ax[4])
    sns.violinplot(data=df[["NKG7", "Cell Type", "Condition"]], x="NKG7", y="Cell Type", hue="Condition",split=True, ax=ax[5])

    
     
    # sns.boxplot(data=weightedProjDF, x= "Cmp. 12", y= "Cell Type", ax=ax[5])
    
    # df1 = dataDF[(dataDF["Condition"] == "CTRL1")]
    # df2 = dataDF[(dataDF["Condition"] == "Loratadine")]
    
    # df = pd.concat([df1, df2])
    
    # sns.boxplot(data=df[["GNLY", "Cell Type", "Condition"]], x="GNLY", y="Cell Type", hue="Condition", ax=ax[6])
    
    
    # sns.boxplot(data=weightedProjDF, x= "Cmp. 20", y= "Cell Type", ax=ax[7])
    
    # df1 = dataDF[(dataDF["Condition"] == "Dexrazoxane HCl (ICRF-187, ADR-529)")]
    # df2 = dataDF[(dataDF["Condition"] == "CTRL1")]
    
    # df = pd.concat([df1, df2])
    
    # sns.boxplot(data=df[["VPREB3", "Cell Type", "Condition"]], x="VPREB3", y="Cell Type", hue="Condition", ax=ax[8])
  
  
  


    return f