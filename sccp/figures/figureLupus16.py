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
from ..figures.commonFuncs.plotFactors import bot_top_genes


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((25, 20), (5, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    cmp=27
    genes = bot_top_genes(X, cmp, geneAmount=3)

    for i, gene in enumerate(np.ravel(genes)):
        plotGenePerStatus(X, gene, ax[i], cellType="leiden")

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


def plot2GenePerCellTypeStatus(
    X, gene1, gene2, celltype1, celltype2, ax, cellType="Cell Type"
):
    """Plots average gene expression across cell types for a category of drugs"""
    gene = [gene1, gene2]

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

        if i == 0:
            df_ = df.copy()
        else:
            df = pd.concat([df_, df]).reset_index()

    df = df.pivot(
        index=["Status", "Cell Type", "Condition"],
        columns="Gene",
        values="Average Gene Expression",
    ).reset_index()
    df1 = df.loc[df["Cell Type"] == celltype1]
    df2 = df.loc[df["Cell Type"] == celltype2]
    df = pd.concat([df1, df2]).reset_index()

    sns.scatterplot(
        data=df,
        x=gene[0],
        y=gene[1],
        hue="Status",
        ax=ax,
    )

    ax.set_title("Average Gene Expression Per Patient")
    ax.set_xlabel(f"{celltype1}: {gene1}")
    ax.set_ylabel(f"{celltype2}: {gene2}")
