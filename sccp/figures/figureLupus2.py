"""
Lupus: UMAP labeled by cell type
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotUMAP import plotLabelsUMAP
import seaborn as sns
import pandas as pd
import numpy as np

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 10), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    X = openPf2(rank, "Lupus")
    
    # print(np.shape(X.X))
    print(np.max(X.X))
    
    # df = X[:, "PPBP"].X
    # print(np.shape(df[(df > 2)]))
    
    # X = X[(X[: , "PPBP"].X > 0.5).flatten(), :] 
    # adata[(adata[:,'elav'].X>0).flatten(), : ] works
    # print(X)
    # print(np.shape(X.X))

    # plotLabelsUMAP(X, "Cell Type", ax[0])
    
    
    
    # genes = ["PF4", "SDPR", "GNG11", "PPBP"]
    
    # for i, cohort in enumerate(["1.0", "2.0", "3.0", "4.0"]):
    #     for j, gene in enumerate(genes)
    
    # print(X.obs["Processing_Cohort"])
    # # print(X.obs["Processing_Cohort"].flatten())
    # print(X.obs["Processing_Cohort"].values)
    # # # for
    # print
    df = pd.DataFrame(data=np.vstack((X[:, "GZMH"].X.flatten(), X.obs["Processing_Cohort"].values)).transpose(), columns=["GZMH", "Cohort"])
    print(df)
    
    # df["Cohort"] = df["Cohort"].replace({1.0: "A", 2.0: "B", 3.0: "C", 4.0: "D"})
    # print(df)
    sns.histplot(df, ax=ax[0], x="GZMH", hue="Cohort", bins=100)
    df2 = pd.DataFrame(data=np.vstack((X[:, "FHIT"].X.flatten(), X.obs["Processing_Cohort"].values)).transpose(), columns=["FHIT", "Cohort"])
    # print(df)
    
    # df2["Cohort"] = df2["Cohort"].replace({1.0: "A", 2.0: "B", 3.0: "C", 4.0: "D"})

    # sns.histplot(p.X[:, "FHIT"].X, ax=ax[0], hue=X.obs["Processing_Cohort"].values, bins=100)
    # # sns.histplot(X[:, "GNG11"].X, ax=ax[1], bins=100)
    sns.histplot(df2, ax=ax[1], x="FHIT", hue="Cohort", bins=100)
    
    
    df3 = pd.DataFrame(data=np.vstack((X[:, "ISG15"].X.flatten(), X.obs["Processing_Cohort"].values)).transpose(), columns=["ISG15", "Cohort"])
    # print(df)
    
    # df3["Cohort"] = df3["Cohort"].replace({1.0: "A", 2.0: "B", 3.0: "C", 4.0: "D"})
    
    sns.histplot(df3, ax=ax[2], x="ISG15", hue="Cohort", bins=100)
    # # sns.histplot(X[:, "PPBP"].X, ax=ax[2], bins=100)
    
    
    # df4 = pd.DataFrame(data=np.vstack((X[:, "SDPR"].X.flatten(), X.obs["Processing_Cohort"].values)).transpose(), columns=["SDPR", "Cohort"])
    # print(df3)
    
    # df4["Cohort"] = df4["Cohort"].replace({1.0: "A", 2.0: "B", 3.0: "C", 4.0: "D"})
    # sns.histplot(df4, ax=ax[3], x="SDPR", hue="Cohort", bins=100)
    # # sns.histplot(X[:, "SDPR"].X, ax=ax[3], bins=100)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')
    ax[3].set_yscale('log')
    # ax[0].set(xlabel="Gene Expression", title="PF4")
    # ax[1].set(xlabel="Gene Expression", title="GNG11")
    # ax[2].set(xlabel="Gene Expression", title="PPBP")
    # ax[3].set(xlabel="Gene Expression", title="SDPR")
    # sns.histplot(X[:, "SDPR"].X, ax=ax[3])

    return f
