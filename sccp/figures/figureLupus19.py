"""
Lupus
"""
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import linregress
import scanpy as sc
from ..imports import import_lupus
from ..figures.commonFuncs.plotFactors import bot_top_genes


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((21, 4), (1, 6))

    # Add subplot labels
    subplotLabel(ax)
    
    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
 
    cmp=22
    amount=100
    downgenes, upgenes = myeloid_score(X)
    
    genes = bot_top_genes(X, cmp, geneAmount=amount)
    
    topgenes = genes[amount:]
    botgenes = genes[:amount]
    
    mask = np.in1d(downgenes, topgenes)
    print(np.sum(mask))
    mask = np.in1d(downgenes, botgenes)
    print(np.sum(mask))
    mask = np.in1d(upgenes, topgenes)
    print(np.sum(mask))
    
    total=np.sum(mask)
    perc = 100* np.sum(mask)/len(upgenes)
    print(perc)
    mask = np.in1d(upgenes, botgenes)
    print(np.sum(mask))
    
    
  
    
    return f


def myeloid_score(X):  
    """Scanpy average gene score for all cells"""
    # Based on DEG analysis for general cell types
    df = pd.read_csv("sccp/data/Lupus/DEgenes.csv")
    df = pd.DataFrame(data=df.iloc[:, [0, -1]].to_numpy(), columns = ["Gene", "Category"])
    df_up = df.loc[df["Category"] == "Up-Monocyte"]
    myeloid_up = df_up["Gene"].to_numpy()
    
    df_down = df.loc[df["Category"] == "Down-Monocyte"]
    myeloid_down = df_down["Gene"].to_numpy()
    
    
    # X = sc.tl.score_genes(adata=X, gene_list=myeloid_genes, copy=True, use_raw=False)

    return myeloid_down, myeloid_up
        
   
def plotScore(X, ax, cellType="Cell Type"):
    """Plots average score  across cell types and patient status"""
    df = pd.DataFrame({"Score": X.obs["score"].values})
    df["Status"] = X.obs["SLE_status"].values
    df["Condition"] = X.obs["Condition"].values
    df["Cell Type"] = X.obs[cellType].values
    df = df.groupby(["Status", "Cell Type", "Condition"], observed=False).mean().reset_index()
    
    sns.boxplot(
        data=df,
        x="Cell Type",
        y="Score",
        hue="Status",
        order=np.unique(df["Cell Type"]),
        ax=ax,
        showfliers=False)
    
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)