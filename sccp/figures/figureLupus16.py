"""
Lupus investigation
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotUMAP import plotLabelsUMAP
import seaborn as sns
import pandas as pd
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((30, 26), (4, 4))

    # Add subplot labels
    subplotLabel(ax)
    
    X = openPf2(rank=40, dataName="Lupus")
    genes = ["PF4", "SDPR", "GNG11", "PPBP"]
    typeofbatch = ["Processing_Cohort", "pool"]
    status = ["SLE", "Healthy"]
    axs = 0

    
    genesV = X[:, genes]
    dataDF = genesV.to_df()
    
    dataDF["Condition"] = genesV.obs["Condition"].values
    dataDF["Louvain"] = genesV.obs["louvain"].values
    dataDF["Status"] = genesV.obs["SLE_status"].values

    print(dataDF)
        
    for i, gene in enumerate(genes):
        for j, batch in enumerate(typeofbatch):
            for k, stat in enumerate(status):
                dataDF["Batch"] = genesV.obs[batch].values
                df = dataDF.loc[dataDF["Louvain"] == "14"]
                print(df)
                # print(df)
                # sns.histplot(data=df.loc[df["Status"] == stat], ax=ax[axs], bins=100, x=gene, hue="Batch", multiple="stack")
                sns.histplot(data=df.loc[df["Status"] == stat], ax=ax[axs], bins=100, x=gene, hue="Batch", multiple="stack", 
                            element="step", fill=False)
                ax[axs].set_yscale('log')
                ax[axs].set(title=f"{gene}:{batch}:{stat}")
                axs+=1
    
    
    
    # for i, gene in enumerate(genes):
    #     for j, batch in enumerate(typeofbatch):
    #         for k, stat in enumerate(status):
    #             df = pd.DataFrame(data=np.vstack((X[:, gene].X.flatten(), X.obs[batch].values)).transpose(), columns=[gene, "Batch"])
    #             df["Status"] = X.obs["SLE_status"].values
    #             sns.histplot(data=df.loc[df["Status"] == stat], ax=ax[axs], bins=100, x=gene, hue="Batch", multiple="stack", 
    #                         element="step", fill=False)
    #             ax[axs].set_yscale('log')
    #             ax[axs].set(title=f"{gene}:{batch}:{stat}")
    #             axs+=1
        

    
    return f
