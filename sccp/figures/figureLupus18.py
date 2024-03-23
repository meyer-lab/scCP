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

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 6), (2, 3))


    # Add subplot labels
    subplotLabel(ax)
    
    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    X = cytotoxic_score(X)
    plotScore(X, ax[0], cellType="Cell Type2")
    
    # X = interferon_score(X)
    # plotScore(X, ax[1], cellType="Cell Type2")
    
    # X = interferon_score(X, csv=False)
    # plotScore(X, ax[2], cellType="Cell Type2")



    # XX = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    # X = import_lupus(); change gene threshold.001
    # X.obs["Cell Type2"] = XX.obs["Cell Type2"]
    # X = cytotoxic_score(X)
    # plotScore(X, ax[3], cellType="Cell Type2")
    
    # X = interferon_score(X)
    # plotScore(X, ax[4], cellType="Cell Type2")
    
    # X = interferon_score(X, csv=False)
    # plotScore(X, ax[5], cellType="Cell Type2")
    
    
    return f


def cytotoxic_score(X):
    """Scanpy avergage gene score for all cells"""
    cytotoxic_genes = ["PRF1", "GZMH", "GZMB"]
    X = sc.tl.score_genes(adata=X, gene_list=cytotoxic_genes, copy=True, use_raw=False)

    return X


def interferon_score(X, csv=True):
    """Scanpy avergage gene score for all cells"""
    if csv is True:
        # Based on DEG analysis for general cell types
        df = pd.read_csv("sccp/data/Lupus/DEgenes.csv")
        df = pd.DataFrame(data=df.iloc[:, [0, -1]].to_numpy(), columns = ["Gene", "Category"])
        df = df.loc[df["Category"] == "Up-Monocyte"]
        interferon_genes = df["Gene"].to_numpy()
    else:
        # Obtained from the figure co-expression figure
        interferon_genes = ["ISG15", "RGS1", "EPSTI1", "CD8A", "IFI16", "STAT1", "HBB", "SP110", "SP100", "EIF2AK2", "IFI27", "IFI44", "IFI44L", "PATL2", "RNF213", "IFI6", "ISG20", "LAG3", "MX1", "XAF1", "BST2", "OAS1"]
        
    X = sc.tl.score_genes(adata=X, gene_list=interferon_genes, copy=True, use_raw=False)

    return X

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
        ax=ax,
        showfliers=False)
    
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)
