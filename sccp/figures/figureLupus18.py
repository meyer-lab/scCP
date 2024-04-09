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
from scipy.stats import linregress, pearsonr, spearmanr

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((5, 5), (2, 2))


    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    dfWPGene(X, 27, "RETN", "44", "leiden", ax[0])
    dfWPGene(X, 27, "S100A8", "44", "leiden", ax[1])
    dfWPGene2(X, 28, "IFI27", "30", "leiden", ax[2], "21")
    # dfWPGene(X, 27, "S100A8", "44", "leiden", ax[1])?
    

    return f

def dfGenePerStatus(X, gene, cellType="Cell Type"):
    """Plots average gene expression across cell types for a category of drugs"""
    genesV = X[:, gene]
    dataDF = genesV.to_df()
    dataDF["Status"] = genesV.obs["SLE_status"].values
    dataDF["Condition"] = genesV.obs["Condition"].values
    dataDF["Cell Type"] = genesV.obs[cellType].values


    df = pd.melt(
        dataDF, id_vars=["Status", "Cell Type", "Condition"], value_vars=gene
    ).rename(columns={"variable": "Gene", "value": "Value"})

    df = df.groupby(["Status", "Cell Type", "Gene", "Condition"], observed=True).mean()
    df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

    return df


def dfWPPerStatus(X, cmp, cellType="Cell Type"):
    """Plots average gene expression across cell types for a category of drugs"""
    dataDF = pd.DataFrame(data=X.obsm["weighted_projections"][:, cmp-1], columns=["WP"])
    dataDF["Status"] = X.obs["SLE_status"].values
    dataDF["Condition"] = X.obs["Condition"].values
    dataDF["Cell Type"] = X.obs[cellType].values


    df = pd.melt(
        dataDF, id_vars=["Status", "Cell Type", "Condition"], value_vars="WP"
    ).rename(columns={"variable": "WP", "value": "WProjs"})

    df = df.groupby(["Status", "Cell Type", "WP", "Condition"], observed=True).mean().reset_index()

    return df



def dfWPGene(X, cmp, gene, cluster, celltype, ax, cluster2=None):
    dfGene = dfGenePerStatus(X, gene, cellType=celltype)
    dfWP = dfWPPerStatus(X, cmp, cellType=celltype)
    dfWP["Gene"] = dfGene["Average Gene Expression"]

    
    louvainclusters = np.unique(dfWP["Cell Type"])
    result = np.where(louvainclusters==cluster)[0]
    louvainclusters = np.delete(louvainclusters,result)
    
    dfWP = dfWP.replace(louvainclusters, f"Other")
    dfWP = dfWP.rename(columns={"Cell Type": "Leiden Cluster"})

    sns.scatterplot(dfWP, x="WProjs", y="Gene", hue="Leiden Cluster", ax=ax)
    ax.set_yscale("log")
    ax.set(ylabel = f"Average {gene} Expression", xlabel="Average Weighted Projs")


def dfWPGene2(X, cmp, gene, cluster, celltype, ax, cluster2=None):
    dfGene = dfGenePerStatus(X, gene, cellType=celltype)
    dfWP = dfWPPerStatus(X, cmp, cellType=celltype)
    dfWP["Gene"] = dfGene["Average Gene Expression"]

    
    louvainclusters = np.unique(dfWP["Cell Type"])
    print(np.where(louvainclusters==cluster)[0])
    print(np.where(louvainclusters==cluster2)[0])
    result = np.concatenate((np.where(louvainclusters==cluster)[0], np.where(louvainclusters==cluster2)[0]))
    louvainclusters = np.delete(louvainclusters,result)
    
    dfWP = dfWP.replace(louvainclusters, f"Other")
    dfWP = dfWP.rename(columns={"Cell Type": "Leiden Cluster"})

    sns.scatterplot(dfWP, x="WProjs", y="Gene", hue="Leiden Cluster", ax=ax)
    ax.set_yscale("log")
    ax.set(ylabel = f"Average {gene} Expression", xlabel="Average Weighted Projs")
