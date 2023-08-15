"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (subplotLabel, getSetup, 
                     plotCmpUMAP, openPf2, 
                     flattenData, flattenWeightedProjs, 
                     plotCellTypeUMAP, plotCmpPerCellType, 
                     plotGenePerCellType)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..imports.gating import gateThomsonCells
import umap 
import os
from os.path import join
import pandas as pd

import seaborn as sns

path_here = os.path.dirname(os.path.dirname(__file__))
import pandas as pd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (4, 4))

    # Add subplot labels
    subplotLabel(ax)
    
    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    dataDF = flattenData(data)
    
    rank = 30
    dataDF["Cell Type"] = gateThomsonCells(rank=rank, saveCellTypes=False)

    # _, factors, projs = openPf2(rank, "Thomson")
    # pf2Points = umap.UMAP(random_state=1).fit(projs)
    
    # # plotCellTypeUMAP(pf2Points, dataDF, ax[0])
        
    # weightedProjDF = flattenWeightedProjs(data, factors, projs)
    # weightedProjDF["Cell Type"] = dataDF["Cell Type"].values
    # weightedProjDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    # dataDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    
    # comps = [5, 6]
    # for i, comp in enumerate(comps):
    #     plotCmpPerCellType(weightedProjDF, comps[i], ax[i+1])

    genes = ["NKG7", "GNLY"]
#     df = dataDF[(dataDF["Condition"].isin(["Budesonide", "CTRL3"]))]
    
    data = pd.melt(dataDF, id_vars=["Condition", "Cell Type"], value_vars=genes).rename(
            columns={"variable": "Gene", "value": "Value"})
    
    
    df = data.groupby(["Condition", "Cell Type", "Gene"]).mean()
    
    print(df)
    
#     .reset_index(name="Mean Expression Across Drugs")


    
    
    sns.stripplot(data=df, x="Gene", y="Value", hue="Cell Type", dodge=True, jitter=False, ax=ax[0])
    # for i, gene in enumerate(genes):
    #     plotGenePerCellType(dataDF[(dataDF["Condition"].isin(["Budesonide", "CTRL3"]))], gene, ax[(2*i)+5])
    #     plotGenePerCellType(dataDF[(dataDF["Condition"].isin(["Triamcinolone Acetonide", "CTRL1"]))], gene, ax[(2*i)+6])
    
    
    return f