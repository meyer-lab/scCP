"""
Lupus: 
"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
import pandas as pd
from .commonFuncs.plotFactors import bot_top_genes
from matplotlib.axes import Axes
import anndata


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((5, 2), (1, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    genes = bot_top_genes(X, cmp=22, geneAmount=1)
    genes = ["IFITM3", "APOBEC3A"]
    
    df_total = pd.DataFrame([])

    for i, gene in enumerate(np.ravel(genes)):
        df = avegene_per_status(X, gene, ax[i], cellType="Cell Type2")
        df_total = pd.concat([df, df_total])

    df_total = df_total.pivot(index=["Status", "Cell Type", "Condition"], columns="Gene", values="Average Gene Expression")
    df_mean = df_total.groupby(["Status", "Cell Type"], observed=False).mean().dropna().reset_index()
    df_std = df_total.groupby(["Status", "Cell Type"], observed=False).std().dropna().reset_index()
    
    sns.scatterplot(
        data=df_total,
        x="APOBEC3A",
        y="IFITM3",
        hue="Cell Type",
        style="Status",
        ax=ax[0]
    )
    colors = sns.color_palette("hls", len(np.unique(df_mean["Cell Type"])))
    fmt = ["o", '*']
    
    for i, status in enumerate(np.unique(df_mean["Status"])):
        for j, celltype in enumerate(np.unique(df_mean["Cell Type"])):
            df_mini_mean = df_mean.loc[(df_mean["Status"] == status) & (df_mean["Cell Type"] == celltype)]
            df_mini_std = df_std.loc[(df_std["Status"] == status) & (df_std["Cell Type"] == celltype)]
            ax[1].errorbar(df_mini_mean["APOBEC3A"], df_mini_mean["IFITM3"], xerr=df_mini_std["APOBEC3A"], yerr=df_mini_std["IFITM3"], ls='none', fmt=fmt[i], label=celltype+status, color=colors[j])
            

    ax[1].set(xlabel="Average APOBEC3A",ylabel="Average IFITM3")
    ax[1].legend()
    
    return f


def plot_avegene_per_status(
    X: anndata.AnnData, gene: str, ax: Axes, cellType="Cell Type"
):
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
    
    
def avegene_per_status(
    X: anndata.AnnData, gene: str, ax: Axes, cellType="Cell Type"
):
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
    df = df.rename(columns={"Value": "Average Gene Expression"}).dropna().reset_index()
    
    
    return df 