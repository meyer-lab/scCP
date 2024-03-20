"""
Lupus: Plots all top and bottom genes Pf2 weights
"""
from anndata import read_h5ad
from .common import getSetup
from .commonFuncs.plotGeneral import plotGeneFactors
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((5, 5), (2, 1))

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")

      
    # cmp = [[22, 28], [27, 28], [22, 28]]
    
    # for i, comp in enumerate(cmp):
    #     XX = np.array(X.varm["Pf2_C"])
    #     XX = XX[:, [comp[0], comp[1]]]
        
    #     ax[i].scatter(x=XX[:,0], y=XX[:,1])
    #     ax[i].set(xlabel=f"Cmp: {comp[0]}", ylabel=f"Cmp: {comp[1]}")
    
    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    print(X.varm["Pf2_C"].shape)
    print(X)
    cytotoxic = ["PRF1", "GZMH", "GZMB"]

    X = sc.tl.score_genes(adata=X, gene_list=cytotoxic, copy=True, use_raw=False)
    plotScore(X, ax[0], cellType="Cell Type2")
    ax[0].set_xticklabels(labels=ax[0].get_xticklabels(), rotation=90)
    print(X.obs["score"])
    # print(a)




    return f


def plotScore(X, ax, cellType="Cell Type"):
    """Plots average gene expression across cell types for a category of drugs"""

    df = pd.DataFrame({"Score": X.obs["score"].values})
    df["Status"] = X.obs["SLE_status"].values
    df["Condition"] = X.obs["Condition"].values
    df["Cell Type"] = X.obs[cellType].values

    df = df.groupby(["Status", "Cell Type", "Condition"], observed=False).mean().reset_index()
    print(df)
    # df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

    sns.boxplot(
        data=df,
        x="Cell Type",
        y="Score",
        hue="Status",
        ax=ax,
        showfliers=False,
    )
    
