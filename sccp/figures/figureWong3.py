"""
Figure 5a_e
"""

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.axes import Axes
import gseapy as gp
from gseapy import Biomart
import mygene

from .common import getSetup, subplotLabel
from .commonFuncs.plotFactors import plot_gene_factors
from .commonFuncs.plotGeneral import cell_count_perc_df, rotate_xaxis
from .commonFuncs.plotPaCMAP import plot_gene_pacmap, plot_wp_pacmap
from .figure4e_k import plot_correlation_cmp_cell_count_perc
from scipy import stats


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((15, 6), (4, 2))
    subplotLabel(ax)


    geneAmount = 40
    plot_loadings_pca_partial(ax[0], top=True, PC=1, geneAmount=geneAmount)
    plot_loadings_pca_partial(ax[1], top=False, PC=1, geneAmount=geneAmount)
    
    plot_loadings_pca_partial(ax[2], top=True, PC=2, geneAmount=geneAmount)
    plot_loadings_pca_partial(ax[3], top=False, PC=2, geneAmount=geneAmount)

    X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    plot_gene_factors_partial(6, X, ax[4], top=True, geneAmount=geneAmount)
    plot_gene_factors_partial(22, X, ax[5], top=True, geneAmount=geneAmount)
    plot_gene_factors_partial(28, X, ax[6], top=True, geneAmount=geneAmount)
    plot_gene_factors_partial(1, X, ax[7], top=True, geneAmount=geneAmount)


    return f


def plot_loadings_pca_partial(ax, geneAmount: int = 15, top=True, PC: int = 1):
    """XXX"""
    if PC == 1:
        df = pd.read_csv("loadings_time_series_PC1.csv", dtype=str).rename(columns={"Unnamed: 0": "Gene"}) 
    else:
        df = pd.read_csv("loadings_time_series_PC2.csv", dtype=str).rename(columns={"Unnamed: 0": "Gene"})
        
    df["PC"+str(PC)] = stats.zscore(df["PC"+str(PC)].to_numpy().astype(float))
    df = df.sort_values(by="PC"+str(PC))
    
    if top:
        sns.barplot(
            data=df.iloc[-geneAmount:, :], x="Gene", y="PC"+str(PC), color="k", ax=ax
            
        )
        higly_weighted_genes = df.iloc[-geneAmount:]["Gene"].values
    else:
        sns.barplot(data=df.iloc[:geneAmount, :], x="Gene", y="PC"+str(PC), color="k", ax=ax)
        higly_weighted_genes = df.iloc[:geneAmount]["Gene"].values

    ax.tick_params(axis="x", rotation=90)
    
    return [gene.upper() for gene in higly_weighted_genes], df


    
def plot_gene_factors_partial(
    cmp: int, X, ax, geneAmount: int = 15, top=True
):
    """Plotting weights for gene factors for both most negatively/positively weighted terms"""
    cmpName = f"Cmp. {cmp}"

    df = pd.DataFrame(
        data=X.varm["Pf2_C"][:, cmp - 1], index=X.var_names, columns=[cmpName]
    )
    df = df.reset_index(names="Gene")
    df = df.sort_values(by=cmpName)

    if top:
        sns.barplot(
            data=df.iloc[-geneAmount:, :], x="Gene", y=cmpName, color="k", ax=ax
        )
    else:
        sns.barplot(data=df.iloc[:geneAmount, :], x="Gene", y=cmpName, color="k", ax=ax)

    ax.tick_params(axis="x", rotation=90)
    
