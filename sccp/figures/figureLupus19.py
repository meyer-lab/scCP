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
from .figureLupus16 import plotGenePerStatus
from .figureLupus18 import getCellCountDF

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 18), (6, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    
    cmp=28
    genes = top_bot_genes(X, cmp=cmp, geneAmount=5)
    
    gated, nongated = cmpGatedDF(X, cmp, perc=5, positive=True)
    
    gatedCount = getCellCountDF(gated, celltype="Cell Type2", cellPerc=False)
    sns.boxplot(data=gatedCount, x="Cell Type", y="Count", hue="SLE_status", ax=ax[0])
    rotateaxis(ax[0])
    nongatedCount = getCellCountDF(nongated, celltype="Cell Type2", cellPerc=False)
    sns.boxplot(data=nongatedCount, x="Cell Type", y="Count", hue="SLE_status", ax=ax[1])
    rotateaxis(ax[1])
   
    
    
    for i, gene in enumerate(np.ravel(genes)):
        df = plotGenePerStatus(gated, gene, ax[(2*i)+2], cellType="Cell Type2")
        ax[(2*i)+2].set(title=f"Gated: {gene}")
        df2 = plotGenePerStatus(nongated, gene, ax[(2*i)+3], cellType="Cell Type2")
        ax[(2*i)+3].set(title=f"NonGated: {gene}")
        rotateaxis(ax[(2*i)+2])
        rotateaxis(ax[(2*i)+3])

    return f


def cmpGatedDF(X, cmp, perc=10, positive=True):
    wProjs = X.obsm["weighted_projections"][:, cmp-1]
    if positive is True:
        threshold = np.percentile(wProjs, 100-perc) # Top Perc#
        idx = np.ravel(np.argwhere(wProjs > threshold))
        nonidx = np.ravel(np.argwhere(wProjs < threshold))
    else: 
        threshold = np.percentile(wProjs, perc) # Bottom Perc#
        idx = np.ravel(np.argwhere(wProjs < threshold))
        nonidx = np.ravel(np.argwhere(wProjs > threshold))
        
    thresholdDF = X[idx, :]
    nonThresholdDF = X[nonidx, :]
    
    return thresholdDF, nonThresholdDF
    
    
def rotateaxis(ax):
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)
    
# def compare(X, celltype="Cell Type", cellPerc=True):
    
#     df = X.obs[["Cell Type", "SLE_status", "Condition", "Cell Type2"]].reset_index(drop=True)

#     dfCond = (
#         df.groupby(["Condition"], observed=True).size().reset_index(name="Cell Number")
#     )
#     dfCellType = (
#         df.groupby([celltype, "Condition", "SLE_status"], observed=True)
#         .size()
#         .reset_index(name="Count")
#     )
#     dfCellType["Count"] = dfCellType["Count"].astype("float")
    
#     if cellPerc is True:
#         for i, cond in enumerate(np.unique(df["Condition"])):
#             dfCellType.loc[dfCellType["Condition"] == cond, "Count"] = (
#                 100
#                 * dfCellType.loc[dfCellType["Condition"] == cond, "Count"].to_numpy()
#                 / dfCond.loc[dfCond["Condition"] == cond]["Cell Number"].to_numpy()
#             )
#         dfCellType.rename(columns={"Count": "Cell Type Percentage"}, inplace=True)
        
#     dfCellType.rename(columns={celltype: "Cell Type"}, inplace=True)
    
#     return dfCellType
    




    
