"""
Figure 5a_e
"""

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.axes import Axes

from .common import getSetup, subplotLabel
from .commonFuncs.plotFactors import plot_gene_factors
from .commonFuncs.plotGeneral import cell_count_perc_df, rotate_xaxis
from .commonFuncs.plotPaCMAP import plot_gene_pacmap, plot_wp_pacmap
from .figure4e_k import plot_correlation_cmp_cell_count_perc


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (3, 3))
    subplotLabel(ax)


    plot_loadings_pca(ax[0])
    # X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    # plot_gene_factors(X, ax[2])

    # plot_wp_pacmap(X, 4, ax[0], 0.25)
    # plot_wp_pacmap(X, 27, ax[1], 0.25)

    # X = cytotoxic_score(X)
    # plot_score(X, ax[3], cellType="Cell Type2")
    # ax[0].set(ylabel="Cytotoxic Score")

    # plot_gene_pacmap("RETN", "Pf2", X, ax[4])

    # celltype_count_perc_df = cell_count_perc_df(X, celltype="leiden", status=True)
    # plot_correlation_cmp_cell_count_perc(
    #     X, 28, celltype_count_perc_df, ax[5], cellPerc=False
    # )

    return f


def cytotoxic_score(X: anndata.AnnData):
    """Scanpy average gene score for all cells"""
    cytotoxic_genes = ["PRF1", "GZMH", "GZMB"]
    X = sc.tl.score_genes(adata=X, gene_list=cytotoxic_genes, copy=True, use_raw=False)

    return X


def plot_score(X: anndata.AnnData, ax: Axes, cellType: str = "Cell Type"):
    """Plots average score  across cell types and patient status"""
    df = pd.DataFrame({"Score": X.obs["score"].values})
    df["Status"] = X.obs["SLE_status"].values
    df["Condition"] = X.obs["Condition"].values
    df["Cell Type"] = X.obs[cellType].values
    df_mean_score = (
        df.groupby(["Status", "Cell Type", "Condition"], observed=False)
        .mean()
        .reset_index()
        .dropna()
        .sort_values(["Cell Type", "Condition"])
    )

    df_count = (
        df.groupby(["Cell Type", "Condition"], observed=False)
        .size()
        .reset_index(name="Cell Count")
        .sort_values(["Cell Type", "Condition"])
    )

    df_mean_score["Cell Count"] = df_count["Cell Count"].to_numpy()

    sns.boxplot(
        data=df_mean_score,
        x="Cell Type",
        y="Score",
        hue="Status",
        order=np.unique(df["Cell Type"]),
        ax=ax,
        showfliers=False,
    )

    rotate_xaxis(ax)


def plot_loadings_pca(ax):
    """Plot GSEA results"""
    df = pd.read_csv("loadings_time_series_PC1.csv", dtype=str).rename("Unnamed: 0", "Gene")
    print(df)
    # df = df.drop(columns=["ID", "Verbose ID"])
    # category = df["Category"].to_numpy().astype(str)

    # df = df.drop(columns=["Category"])
    # df["Process"] = category
    # df = df.iloc[:1000, :]
    # df["Total Genes"] = df.iloc[:, 2:-1].astype(int).sum(axis=1).to_numpy()
    # df = df.loc[df.loc[:, "Process"] == "GO: Biological Process"]
    # df["pValue"] = df["pValue"].astype(float)

    # sns.scatterplot(
    #     data=df.iloc[:10, :], x="pValue", y="Name", hue="Total Genes", ax=ax
    # )
    # ax.set(xscale="log")