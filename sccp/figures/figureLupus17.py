"""
Thomson: Plotting normalized genes and separating data by status (and celltype)
"""
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
import pandas as pd
from .figureCITEseq5 import top_bot_genes
from .commonFuncs.plotUMAP import plotCmpGeneWeightedUMAP, plotCmpGeneWeightedPerCellType

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 18), (6, 4))


    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    df = X.obs[["Cell Type", "SLE_status", "Condition", "Cell Type2"]].reset_index(drop=True)

    dfCond = (
        df.groupby(["Condition"], observed=True).size().reset_index(name="Cell Number")
    )
    dfCellType = (
        df.groupby(["Cell Type2", "Condition", "SLE_status"], observed=True)
        .size()
        .reset_index(name="Count")
    )
    dfCellType["Count"] = dfCellType["Count"].astype("float")
    dfCellType2 = dfCellType.copy()
    
    for i, cond in enumerate(np.unique(df["Condition"])):
        dfCellType.loc[dfCellType["Condition"] == cond, "Count"] = (
            100
            * dfCellType.loc[dfCellType["Condition"] == cond, "Count"].to_numpy()
            / dfCond.loc[dfCond["Condition"] == cond]["Cell Number"].to_numpy()
        )
        
    dfCellType.rename(columns={"Count": "Cell Type Percentage"}, inplace=True)

    
    
    yt = np.unique(X.obs["Condition"])
    factorsA = np.array(X.uns["Pf2_A"])
    factorsA = factorsA[:, 0]
    
    condNames, idx = np.unique(dfCellType["Condition"], return_index=True)
 
    
    totaldf = pd.DataFrame([])

    # dfCellType["Condition"] = pd.Categorical(dfCellType["Condition"], yt)
    
    for i, celltype in enumerate(np.unique(df["Cell Type2"])):
        for j, cond in enumerate(np.unique(df["Condition"])):
            status = np.unique(dfCellType.loc[dfCellType["Condition"] == cond]["SLE_status"])
            smalldf = dfCellType.loc[(dfCellType["Condition"] == cond) & (dfCellType["Cell Type2"] == celltype)]
        
            # Cell Type Percentage
            print(smalldf.empty)
            
            if smalldf.empty is False: 
                # smalldf["Cmp"] = factorsA[j]
                smalldf = smalldf.assign(Cmp=factorsA[j])
            else:
                smalldf = pd.DataFrame({"Condition": cond, "Cell Type2": celltype, "SLE_status": status,
                                                  "Cell Type Percentage": 0, "Cmp": factorsA[j]})
            
            totaldf = pd.concat([totaldf, smalldf])
            
            
            print(totaldf)
        if celltype == "Progen":
            break
        
        
            
    
            
            # totaldf = pd.concat([totaldf, pd.DataFrame({"Condition": cond, "Cell Type": celltype, "Status": smalldf["SLE_status"].to_numpy(),
            #                                             "Cell Type Percentage": smalldf["Cell Type Percentage"].to_numpy()"})])
            
            
        
            
        # dfCellTypeSpecific = dfCellType.loc[dfCellType["Cell Type2"] == celltype]
        # dfCellTypeSpecific = dfCellTypeSpecific.sort_values(by=["Condition"])
        # print(celltype)
        # print(dfCellTypeSpecific)
        
      

        # assert yt.all() == dfCellTypeSpecific["Condition"].to_numpy().all()
        
        
    #     array = []
    #     if celltype == "Progen":
    #         smallCond = dfCellTypeSpecific["Condition"].to_numpy()
    #         for i in range(len(smallCond)):
                
    #             # idx = np.where(np.any(yt == smallCond[i]))
    #             # print(idx)
                
    #             # print(np.asarray(yt == smallCond[i]))
    #             idx = np.asarray(yt == smallCond[i]).nonzero()
           
            
    #             # print(idx[0].flatten())
    #             # # a
    #             array = np.append(array, idx[0])
            
       
                
                
                
                
    #             # [i for i in range(len(a)) if a[i] > 2]
    #             # if smallCond[i] == yt[i]:
    #             #     array.append(True)
    #             # else:
    #             #     array.append(False)
                
    #         array = array.astype(int)
            
    #         dfCellTypeSpecific = dfCellTypeSpecific.iloc[idx, :]
    #      dfCellTypeSpecific["Cmp"] = factorsA.astype("float")
    #     dfCellTypeSpecific["Cmp"] = dfCellTypeSpecific["Cmp"].astype("float")
            
            
    # else: 
    #     dfCellTypeSpecific = dfCellTypeSpecific.iloc[idx, :]
    #     dfCellTypeSpecific["Cmp"] = factorsA.astype("float")
    #     dfCellTypeSpecific["Cmp"] = dfCellTypeSpecific["Cmp"].astype("float")
      
    #     # print(np.shape(dfCellTypeSpecific))
  
            

        
        # sns.scatterplot(data=dfCellTypeSpecific, x="Cmp", y="Cell Type Percentage", hue="SLE_status", ax=ax[i])
        

    
  

    return f
