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


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((25, 20), (5, 6))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    
    
    # genes = [["IFI27", "CCR7"]]
    genes = top_bot_genes(X, cmp=27, geneAmount=25)
    
    for i in genes:
        print(i)
    print("bot", genes[0:25])
    print("top", genes[25:])
    # genes2 = top_bot_genes(X, cmp=14, geneAmount=5)


    # for i, gene in enumerate(np.ravel(genes)):
    #     plotGenePerStatus(X, gene, ax[i], cellType="Cell Type2")
    #     ax[i].set_xticklabels(labels=ax[i].get_xticklabels(), rotation=90)

    # for i in range(len(genes1)):
    #     plot2GenePerCellTypeStatus(X, genes1[i], genes2[-i-1], "T4 Naive", "T4 EM", ax[i], cellType="Cell Type2")


    return f


def plotGenePerStatus(X, gene, ax, cellType="Cell Type"):
    """Plots average gene expression across cell types for a category of drugs"""
    genesV = X[:, gene]
    dataDF = genesV.to_df()
    dataDF = dataDF.subtract(genesV.var["means"].values)
    dataDF["Status"] = genesV.obs["SLE_status"].values
    dataDF["Condition"] = genesV.obs["Condition"].values
    dataDF["Cell Type"] = genesV.obs[cellType].values

    df = pd.melt(
        dataDF, id_vars=["Status", "Cell Type", "Condition"], value_vars=gene
    ).rename(columns={"variable": "Gene", "value": "Value"})


    df = df.groupby(["Status", "Cell Type", "Gene", "Condition"], observed=False).mean()
    df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

    sns.boxplot(
        data=df.loc[df["Gene"] == gene],
        x="Cell Type",
        y="Average Gene Expression",
        hue="Status",
        ax=ax,
        showfliers=False,
    )
    ax.set(title=gene)
    
    return df


def plot2GenePerCellTypeStatus(
    X, gene1, gene2, celltype1, celltype2, ax, cellType="Cell Type"
):
    """Plots average gene expression across cell types for a category of drugs"""
    gene = [gene1, gene2]
    celltype = [celltype1, celltype2]

    for i in range(2):
        genesV = X[:, gene[i]]
        dataDF = genesV.to_df()
        dataDF = dataDF.subtract(genesV.var["means"].values)
        dataDF["Status"] = genesV.obs["SLE_status"].values
        dataDF["Condition"] = genesV.obs["Condition"].values
        dataDF["Cell Type"] = genesV.obs[cellType].values

        df = pd.melt(
            dataDF, id_vars=["Status", "Cell Type", "Condition"], value_vars=gene[i]
        ).rename(columns={"variable": "Gene", "value": "Value"})

        df = df.groupby(
            ["Status", "Cell Type", "Gene", "Condition"], observed=False
        ).mean()
        df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()
        
        df = df.pivot(
            index=["Status", "Cell Type", "Condition"],
            columns="Gene",
            values="Average Gene Expression",
        ).reset_index().dropna()

        df = df.loc[df["Cell Type"] == celltype[i]].drop(columns=["Cell Type"])
        
        if i == 0:
            df_ = df.copy()
        else:
            df = pd.merge(df_, df)

    sns.scatterplot(
        data=df,
        x=gene1,
        y=gene2,
        hue="Status",
        ax=ax,
    )
    ax.set_title("Average Gene Expression Per Patient")
    ax.set_xlabel(f"{celltype1}: {gene1}")
    ax.set_ylabel(f"{celltype2}: {gene2}")
