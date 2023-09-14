"""
Investigation of raw data for Thomson dataset
"""
from .common import subplotLabel, getSetup, flattenData
from .commonFuncs.plotGeneral import plotCellTypePerExpCount, plotCellTypePerExpPerc
from ..imports.scRNA import ThompsonXA_SCGenes
from ..imports.gating import gateThomsonCells
import pandas as pd
import seaborn as sns


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    # dataDF = flattenData(data)
    # dataDF["Cell Type"] = gateThomsonCells(rank=30, saveCellTypes=False)
    # df = pd.read_csv("sccp/data/Thomson/TopBotGenes_Cmp30.csv").rename(columns={"Unnamed: 0": "Gene"})
    # print(df)
    # cmp = 12

    # cmpName = "Cmp. "+str(cmp)
    # df = df[["Gene", cmpName]].sort_values(by=[cmpName])
    # print(df)
    
    # geneAmount=20
    # genesTop = np.empty((geneAmount, X.shape[1]), dtype="<U10")
    #             genesBottom = np.empty((geneAmount, X.shape[1]), dtype="<U10")
    #             sort_idx = np.argsort(X, axis=0)

    
    #             sortGenes = yt[sort_idx[:, j]]
    #             genesTop[:, j] = np.flip(sortGenes[-geneAmount:])  
    #             genesBottom[:, j] = sortGenes[:geneAmount]

    # print(df)
    # sns.boxplot(data=weightedprojs[[cmpName, "Cell Type"]]
    
    # sns.barplot(data=df.iloc[:20,:], x="Gene", y=cmpName, color="k", ax=ax[0])
    # sns.barplot(data=df.iloc[-20:,:], x="Gene", y=cmpName, color="k", ax=ax[1])
    # ax[0].tick_params(axis="x", rotation=90)
    # ax[1].tick_params(axis="x", rotation=90)
        
    

    # for i, drug in enumerate(data.condition_labels):
    #     if (i <12):
    #         print(i)
    #         plotCellTypePerExpCount(
    #             dataDF.loc[dataDF["Condition"] == drug], drug, ax[2 * (i)]
    #         )
    #         plotCellTypePerExpPerc(
    #             dataDF.loc[dataDF["Condition"] == drug], drug, ax[1 + (2 * (i))]
    #         )

    return f
