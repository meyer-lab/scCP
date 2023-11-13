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
    ax, f = getSetup((30, 26), (5, 5))

    # Add subplot labels
    subplotLabel(ax)
    
    # X = openPf2(rank=40, dataName="Lupus")
    # # genes = ["PF4", "SDPR", "GNG11", "PPBP"]
    # genes = ["PPBP"]
    # # typeofbatch = ["Processing_Cohort", "pool"]
    # typeofbatch = "pool"
    # status = ["SLE", "Healthy"]
    # axs = 0
    
    # # dmx_count_AH7TNHDMXX_YE_8-30
    # # dmx_count_AHCM2CDMXX_YE_0831
    # # dmx_count_BH7YT2DMXX_YE_0907

    
    # genesV = X[:, genes]
    # dataDF = genesV.to_df()
    
    # # dataDF["Condition"] = genesV.obs["Condition"].values
    # dataDF["Louvain"] = genesV.obs["louvain"].values
    # dataDF["Status"] = genesV.obs["SLE_status"].values
    
       
    # for i, gene in enumerate(genes):
    #     dataDF["Batch"] = genesV.obs[typeofbatch].values
    #     data = dataDF.loc[dataDF["Louvain"] == "14"]
    #     for k, batch in enumerate(np.unique(genesV.obs[typeofbatch].values)):
    #         # df = df.loc[df["Status"] == stat]
    #         df = data.loc[data["Batch"] == batch]
            
    #         print(df)
        
    #         sns.histplot(data=df, ax=ax[axs], bins=100, x=gene, hue="Status", element="step", fill=False)
    #         ax[axs].set_yscale('log')
    #         ax[axs].set(title=f"{gene}:{batch}")
    #         axs+=1
                
    #             # return f

                
    

    # print(dataDF)
        
    # for i, gene in enumerate(genes):
    #     for j, batch in enumerate(typeofbatch):
    #         for k, stat in enumerate(status):
    #             dataDF["Batch"] = genesV.obs[batch].values
    #             df = dataDF.loc[dataDF["Louvain"] == "14"]
                
    #             for l, 
    #             # print(df)
    #             # print(df)
    #             # sns.histplot(data=df.loc[df["Status"] == stat], ax=ax[axs], bins=100, x=gene, hue="Batch", multiple="stack")
    #             sns.histplot(data=df.loc[df["Status"] == stat], ax=ax[axs], bins=100, x=gene, hue="Batch", multiple="stack", 
    #                         element="step", fill=False)
    #             ax[axs].set_yscale('log')
    #             ax[axs].set(title=f"{gene}:{batch}:{stat}")
    #             axs+=1
    
    
    
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
        

    X = openPf2(rank=40, dataName="Lupus")
    genes = ["PF4", "SDPR", "GNG11", "PPBP"]
    # genes = ["PF4"]
    # typeofbatch = ["Processing_Cohort", "pool"]
    typeofbatch = "pool"
    status = ["SLE", "Healthy"]
    axs = 0
    
    onlyHealthy = ["dmx_count_AH7TNHDMXX_YE_8-30", "dmx_count_AHCM2CDMXX_YE_0831", "dmx_count_BH7YT2DMXX_YE_0907"]

    
    genesV = X[:, genes]
    dataDF = genesV.to_df()
    
    dataDF["Condition"] = genesV.obs["Condition"].values
    dataDF["Louvain"] = genesV.obs["louvain"].values
    dataDF["Status"] = genesV.obs["SLE_status"].values
    
       
    for i, gene in enumerate(genes):
        dataDF["Batch"] = genesV.obs[typeofbatch].values
        data = dataDF.loc[dataDF["Louvain"] == "14"]
    
            # df = df.loc[df["Status"] == stat]
            # df = data.loc[data["Batch"] == batch]
        df = data.loc[data["Status"] == "Healthy"]
        
        labels = df["Batch"].values
        
        labels = [c if c in onlyHealthy else "HealthyLupus" for c in labels]
        print(np.unique(labels))
        labels = [c if c in "HealthyLupus" else "OnlyHealthy" for c in labels]
        print(np.unique(labels))
        
        df["Batch2"] = labels
        
        print(df)
        # for k, batch in enumerate(np.unique(labels)):
        sns.histplot(data=df, ax=ax[axs], bins=100, x=gene, hue="Batch", element="step", fill=False)
        ax[axs].set_yscale('log')
        ax[axs].set(title=f"{gene}")
        axs+=1
        
     
    
    return f
