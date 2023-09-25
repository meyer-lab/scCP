
"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
    openUMAP,
    flattenData,
    flattenWeightedProjs,
)
from .commonFuncs.plotGeneral import plotGenePerCellType, plotGenePerCategCond
from .commonFuncs.plotUMAP import (
    plotCellTypeUMAP,
    plotCmpPerCellType,
    plotCmpUMAP,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..imports.gating import gateThomsonCells
import pandas as pd
import seaborn as sns 
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (4, 5))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    dataDF = flattenData(data)

    rank = 30
    dataDF["Cell Type"] = gateThomsonCells(rank=rank, saveCellTypes=False)
    
    # geneSet4 = ["CCNB1", "GADD45A", "SLC40A1", "CDC20", "ITIH3", "CPEB1", "CBR3", "MT1H", "MT1G", "PLK1", "FBN1", "CCR7", "TPX2", 
    #             "FAM131C", "MT1M", "PRSS57", "CENPE", "LTA"]





    # geneSet4 = ["VPREB3", "CD79A", "FAM111B", "HOPX", "SLC30A3", "MS4A1", "TYMS", "RRM2", "TESC", "CD70", "BANK1", "CD79B",
    # "KIAA0101", "ZBED2", "CTSW", "CXCL9", "CCNA2", "CXCL10", "MKI67"]



    geneSet4 = ["VPREB3", "FAM111B"]
    plotGenePerCategStatus(["Dexrazoxane HCl (ICRF-187, ADR-529)"], "Dex HCl", geneSet4, dataDF, ax[0:4])
    
    
    geneSet4 = ["VPREB3", "FAM111B"]
    plotGenePerCategCond(["Dexrazoxane HCl (ICRF-187, ADR-529)"], "Dex HCl", geneSet4, dataDF, ax[4:8])


    return f



def plotGenePerCategStatus(conds, categoryCond, genes, dataDF, axs):

    """Plots average gene expression across cell types for a category of drugs"""
    df = pd.melt(dataDF, id_vars=["Condition", "Cell Type"], value_vars=genes).rename(
            columns={"variable": "Gene", "value": "Value"})
    
    
    print(df.groupby(["Condition", "Cell Type", "Gene"]).median())
    
    

    
    df["Condition"] = np.where(df["Condition"].isin(conds), df["Condition"], "Other")
    for i in conds:
        df = df.replace({"Condition": {i: categoryCond}})
        
    
    
    print(df.groupby(["Condition", "Cell Type", "Gene"]).median())
    
    
    for i, gene in enumerate(genes):
        df1 = df.loc[df["Gene"] == gene]
        sns.boxplot(data=df1, x="Cell Type", y="Value", hue="Condition", ax=axs[i])
        axs[i].set(title=gene)
    
    
    
    return



def plotGenePerCategCond(conds, categoryCond, genes, dataDF, axs):
    """Plots average gene expression across cell types for a category of drugs"""
    data = pd.melt(dataDF, id_vars=["Condition", "Cell Type"], value_vars=genes).rename(
            columns={"variable": "Gene", "value": "Value"})
    df = data.groupby(["Condition", "Cell Type", "Gene"]).median()
    
    print(df)
    df = df.rename(columns={"Value": "Average Gene Expression For Drugs"}).reset_index()
    
    df["Condition"] = np.where(df["Condition"].isin(conds), df["Condition"], "Other")
    for i in conds:
        df = df.replace({"Condition": {i: categoryCond}})

    for i, gene in enumerate(genes):
        sns.boxplot(data=df.loc[df["Gene"] == gene], x="Cell Type", y="Average Gene Expression For Drugs", hue="Condition", ax=axs[i])
        axs[i].set(title=gene)
    