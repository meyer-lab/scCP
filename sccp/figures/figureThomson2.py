"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
    openUMAP,
    flattenData,
    flattenWeightedProjs
)
from .commonFuncs.plotGeneral import(
    plotGenePerCellType
)
from .commonFuncs.plotUMAP import (
    plotCellTypeUMAP,
    plotCmpPerCellType,
    plotCmpUMAP
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..imports.gating import gateThomsonCells
import pandas as pd
import seaborn as sns
import numpy as np


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
    # pf2Points = openUMAP(rank, "Thomson", opt=False)

    # plotCellTypeUMAP(pf2Points, dataDF, ax[0])

    # weightedProjDF = flattenWeightedProjs(data, factors, projs)
    # weightedProjDF["Cell Type"] = dataDF["Cell Type"].values
    # weightedProjDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    # dataDF.sort_values(by=["Condition", "Cell Type"], inplace=True)


    # comps = [5, 12, 20, 30]
    # for i, comp in enumerate(comps):
    #     plotCmpPerCellType(weightedProjDF, comps[i], ax[(2 * i) + 1], outliers=False)
    #     plotCmpUMAP(comps[i], factors, pf2Points, projs, ax[(2 * i) + 2])

    # set1 = ["NKG7", "GNLY", "GZMB", "GZMH", "PRF1"]
    # set2 = ["MS4A1", "CD79A", "CD79B", "TNFRSF13B", "BANK1"]
    # set3 = ["VPREB3", "CD79A", "FAM111B", "HOPX", "SLC30A3", "MS4A1"]
    genes = ["ADORA3", "CD163", "MS4A6A", "MTMR11", "RNASE1"]
    # set4 = ["CCNB1", "GADD45A", "SLC40A1", "CDC20"]
    
    data = pd.melt(dataDF, id_vars=["Condition", "Cell Type"], value_vars=genes).rename(
            columns={"variable": "Gene", "value": "Value"})
    df = data.groupby(["Condition", "Cell Type", "Gene"]).mean()
    
    df = df.rename(columns={"Value": "Average Gene Expression For Drugs"}).reset_index()
    print(df)
    # # df = df.loc[df["Cell Type"] == "B Cells"]
    # drug = "Dexrazoxane HCl (ICRF-187, ADR-529)"
    # drugA = "Dex HCl"
    
    drugs = ["Betamethasone Valerate", "Loteprednol etabonate", "Budesonide", "Triamcinolone Acetonide", "Meprednisone"]
    # for i in drugs:
    #     print(i)
    df["Condition"] = np.where(df["Condition"].isin(drugs), df["Condition"], "Other")
        # df["Condition"]= df["Condition"].where(df["Condition"] == i, "Gluco")
        # df["Condition"]= df["Condition"].where(df["Condition"] == drug, drugA).where(df["Condition"] != drug, "Oth")
        # df["Condition"]= df["Condition"].where(df["Condition"] == i, "Gluco").where(df["Condition"] != drug, "Oth")
    # df["Condition"]= df["Condition"].where(df["Condition"] == drug, "Oth")
    
    # print(np.unique(df["Condition"]))
    # print(df)
    # df["Condition"] = np.where(df["Condition"].isin(["Gluco"]), df["Condition"], "Other")
    
    for i in drugs:
        print(i)
        df = df.replace({"Condition": {i: "Gluco"}})
    
    print(np.unique(df["Condition"]))
    print(df)
    
    
    # df["Condition"]= df["Condition"].where(df["Condition"] == drug, drug).where(df["Condition"] < 60, 'Pass')
    # df["Condition"]= df['score'].where(df['score']  60, "Dexrazoxane HCl (ICRF-187, ADR-529").where(df['score'] < 60, 'Pass')
    # sns.stripplot(data=df.loc[df["Cell Type"] == "B Cells"], x="Gene", y="Average Gene Expression For Drugs", hue="Condition", jitter=False, ax=ax)
    
    # a
    for i, gene in enumerate(genes):
        sns.boxplot(data=df.loc[df["Gene"] == gene], x="Cell Type", y="Average Gene Expression For Drugs", hue="Condition", ax=ax[i])







    # genes = [set1, set2, set3, set4]
    # genes = [set3]
    # for i in range(len(genes)):
    #     plotGenePerCellType(genes[i], dataDF, ax[i + 9])
        
        

    return f
