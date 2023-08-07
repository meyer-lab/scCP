"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup, plotCmpUMAP, openPf2, flattenData
from ..imports.scRNA import ThompsonXA_SCGenes
from parafac2 import parafac2_nd
import umap 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import umap.plot
import os
from os.path import join
from pandas.plotting import parallel_coordinates as pc

path_here = os.path.dirname(os.path.dirname(__file__))

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((30, 30), (4, 5))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    dataDF = flattenData(data)
    
    rank = 30

    weight, factors, projs = openPf2(rank, "Thomson")
    
    # # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1).fit_transform(projs)
    
    cmp=30
    weightedProjs = projs @ factors[1]
    weightedProjs = weightedProjs[:, cmp-1]
       
    # cellSkip = 15
    umap1 = pf2Points[:, 0]
    umap2 = pf2Points[:, 1]
    
    cells = np.zeros(len(umap1))

    df = pd.DataFrame(data={"UMAP1": umap1, "UMAP2": umap2, "Cell Type": cells})
    
    idx = df.index[(df["UMAP1"] >= 5)].tolist()
    df.loc[idx, "Cell Type"] = "DCs"
    
    idx = df.index[(df["UMAP2"] >= 9.5)].tolist()
    df.loc[idx, "Cell Type"] = "B Cells"
    
    
    idx = df.index[(df["UMAP1"] >= -5) & (df["UMAP1"] <= 5) &
                   (df["UMAP2"] >= -3) & (df["UMAP2"] <= 5)].tolist()
    df.loc[idx, "Cell Type"] = "Monocytes"
    
    
    idx = df.index[(df["UMAP1"] <= -.75) &
                   (df["UMAP2"] >= 5) & (df["UMAP2"] <= 9.5)].tolist()
    df.loc[idx, "Cell Type"] = "NK Cells"
    
    
    idx = df.index[(df["UMAP1"] >= -.75) &
                   (df["UMAP2"] >= 5) & (df["UMAP2"] <= 9.5)].tolist()
    df.loc[idx, "Cell Type"] = "T Cells"

    dataName="Thomson"
    np.save(join(path_here, "data/"+dataName+"/"+dataName+"CellTypes.npy"),  df["Cell Type"].values)

    
    # dataDF.to_csv(join(path_here, "data/Thomson/Thomson_DataWCellType.csv"))
    
    
    
    # # idx = dataDF.index[(dataDF["Condition"] == "CTRL1") & (dataDF["Condition"] == "Triamcinolone Acetonide") ].tolist()
    
    # idx = dataDF.index[(dataDF["Condition"] == "CTRL1")].tolist()
    # print(idx)
    # print(dataDF)
    # dataDF = dataDF.loc[idx, :] 
    # for i, drug in enumerate(data.condition_labels):
    #     # sns.violinplot(data=df, x="age", y="class")
    #     if i > 30:
        
    #         print(i)
    #         sns.histplot(data=dataDF.loc[dataDF["Condition"] == drug], x="Cell Type", ax=ax[i-30])
    #         ax[i-30].set(title=drug)
    
    # df1 = dataDF[(dataDF["Condition"] == "Triamcinolone Acetonide")]
    # print(df1.groupby(["Cell Type"]).count())
    # df2 = dataDF[(dataDF["Condition"] == "CTRL1")]
    # print(df2.groupby(["Cell Type"]).count())
    

    
    # df3 = dataDF[(dataDF["Condition"] == "CTRL2")]
    # df4 = dataDF[(dataDF["Condition"] == "CTRL3")]
    # df5 = dataDF[(dataDF["Condition"] == "CTRL4")]
    # df6 = dataDF[(dataDF["Condition"] == "CTRL5")]
    # df7 = dataDF[(dataDF["Condition"] == "CTRL6")]
    
    # dataDF = pd.concat([df1, df7])
    # # dataDF = dataDF[(dataDF["Condition"] == "CTRL1") & (dataDF["Condition"] == "Triamcinolone Acetonide")]
    # print(dataDF)
    # sns.violinplot(data=dataDF, x="CD163", y="Cell Type", hue="Condition", split=True, ax=ax[0])
    # sns.violinplot(data=df1[["CD163", "Cell Type", "Condition"]], x="CD163", hue="Cell Type", ax=ax[0])
    # sns.violinplot(data=df7[["CD163", "Cell Type", "Condition"]], x="CD163", hue="Cell Type", ax=ax[1])
    
    # sns.violinplot(data=dataDF, x="CD163", y="Cell Type", hue="Condition", split=True, ax=ax[1])
    # ax[1].set_xscale('log')
    # sns.boxplot(data=dataDF, x="CD163", y="Cell Type", hue="Condition", ax=ax[0])
    # sns.violinplot(data=pd.concat([df1, df4]), x="CD163", y="Cell Type", hue="Condition", split=True, ax=ax[2])
    # sns.violinplot(data=pd.concat([df1, df5]), x="CD163", y="Cell Type", hue="Condition", split=True, ax=ax[3])
    # sns.violinplot(data=pd.concat([df1, df6]), x="CD163", y="Cell Type", hue="Condition", split=True, ax=ax[4])
    # sns.violinplot(data=pd.concat([df1, df7]), x="CD163", y="Cell Type", hue="Condition", split=True, ax=ax[5])
    # sns.violinplot(data=pd.concat([df1, df2]), x="CD163", y="Cell Type", hue="Condition", split=True, ax=ax[0])

    
    
    # mapper = umap.UMAP(random_state=1).fit(projs)
    
    # cmap = sns.color_palette("Paired")
    
    # umap.plot.points(mapper, labels=df["Cell Type"].values, cmap=cmap, ax=ax[0])
    
    
    
    # nk cells, x less then -.75
    # btween 5 and 9 fory 
    
    
    #macrophagse -5 and 5 for x
    # betwen -2.5 and 4.5 for y 
    
    
    
    # cmap = sns.diverging_palette(240, 10, as_cmap=True)
    # weightedProjs = weightedProjs / np.max(np.abs(weightedProjs))
    # psm = plt.pcolormesh([[-1, 1],[-1, 1]], cmap=cmap)
    
    # weightedProjs =  weightedProjs
    # ax[0].scatter(
    #         umap1,
    #         umap2,
    #         c=weightedProjs,
    #         cmap=cmap,
    #         s=0.2,
    # )
    # ax[0].set_xticks(np.arange(-5, 10, .5))
    # ax[0].set_yticks(np.arange(-3, 12.5, .5))
    # plt.colorbar(psm, ax=ax[0])
    
    # dentridit cells , X>5 
    
    # b cells greater than B 9.5
    
    # nk cells, x less then -.75
    # btween 5 and 9 fory 

    #macrophagse -5 and 5 for x
    # betwen -2.5 and 4.5 for y 

    
    # cellState = np.arange(25, 31, 1) 
    # component = np.copy(cellState)
    
    # cmp = 30
    # weightedProjs = projs @ factors[1]
    # weightedProjs = weightedProjs[:, cmp-1]
    
    # idx = np.argwhere(weightedProjs<0)
    
    # print(idx)
    # print(len(weightedProjs[idx]))


    
    
    
    
    
    # idx = np.argwhere(weightedProjs>0)
    # print(idx)
    # print(len(weightedProjs[idx]))
    
    
    # for i in range(len(cellState)):
    #     plotCmpUMAP(cellState[i], component[i], factors, pf2Points, projs, ax[i])

    return f


