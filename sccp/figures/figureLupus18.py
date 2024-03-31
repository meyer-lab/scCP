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
from ..figures.commonFuncs.plotFactors import bot_top_genes
import scanpy as sc

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((14, 8), (2, 4))


    # Add subplot labels
    subplotLabel(ax)
    # Top .1% for cmp 28 and top .5% for cmp 27
    cmp = 27
    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    X = cmpGated(X, cmp, perc=.5, positive=False)
    
    genes = bot_top_genes(X, cmp, geneAmount=3)
    
    for i, gene in enumerate(genes):
        plotGenePerStatus(X, gene, ax[i], cellType="Cell Type2")
        ax[i].set_xticklabels(labels=ax[i].get_xticklabels(), rotation=90)
        
    X = interferon_score(X)

    plotScore(X, ax[7], cellType="Cell Type2")
        

    return f

def cmpGated(X, cmp, perc=10, positive=True):
    wProjs = X.obsm["weighted_projections"][:, cmp-1]
    if positive is True:
        threshold = np.percentile(wProjs, 100-perc) # Top %
        idx = np.ravel(np.argwhere(wProjs > threshold))
        nonidx = np.ravel(np.argwhere(wProjs < threshold))
    else: 
        threshold = np.percentile(wProjs, perc) # Bottom %
        idx = np.ravel(np.argwhere(wProjs < threshold))
        nonidx = np.ravel(np.argwhere(wProjs > threshold))

    mask = np.repeat("EmptyArray", X.shape[0])
    mask[idx] = "Gated"
    mask[nonidx] = "NonGated"
    X.obs["Gated"] = mask

    return X



def plotGenePerStatus(X, gene, ax, cellType="Cell Type"):
    """Plots average gene expression across cell types for a category of drugs"""
    genesV = X[:, gene]
    dataDF = genesV.to_df()
    dataDF = dataDF.subtract(genesV.var["means"].values)
    dataDF["Status"] = genesV.obs["SLE_status"].values
    dataDF["Condition"] = genesV.obs["Condition"].values
    dataDF["Cell Type"] = genesV.obs[cellType].values
    dataDF["Gated"] = genesV.obs["Gated"].values
    dataDF["Combined"] = dataDF[["Gated", "Status"]].agg(''.join, axis=1)
    dataDF = dataDF.drop(columns=["Gated", "Status"])

    df = pd.melt(
        dataDF, id_vars=["Cell Type", "Condition", "Combined"], value_vars=gene
    ).rename(columns={"variable": "Gene", "value": "Value"})
    
    df = df.groupby(["Cell Type", "Gene", "Combined", "Condition"], observed=False).mean()
    df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

    sns.boxplot(
        data=df.loc[df["Gene"] == gene],
        x="Cell Type",
        y="Average Gene Expression",
        hue="Combined",
        ax=ax,
        showfliers=False,
    )
    ax.set(title=gene)
    
    
    
    
def interferon_score(X, csv=True):
    """Scanpy avergage gene score for all cells"""
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
    df["Gated"] = X.obs["Gated"].values
    df["Combined"] = df[["Gated", "Status"]].agg(''.join, axis=1)
    df = df.drop(columns=["Gated", "Status"])
    
    df = df.groupby(["Combined", "Cell Type", "Condition"], observed=False).mean().reset_index()


    sns.boxplot(
        data=df,
        x="Cell Type",
        y="Score",
        hue="Combined",
        ax=ax,
        showfliers=False)

    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)