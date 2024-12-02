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
from scipy import stats


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (2, 2))
    subplotLabel(ax)


    top_pc1_genes = plot_loadings_pca_partial(ax[0], top=True, PC=1)
    bot_pc1_genes = plot_loadings_pca_partial(ax[1], top=False, PC=1)
    
    top_pc2_genes = plot_loadings_pca_partial(ax[2], top=True, PC=2)
    bot_pc2_genes = plot_loadings_pca_partial(ax[3], top=False, PC=2)
    
    
    
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    genes = [top_pc1_genes, bot_pc1_genes, top_pc2_genes, bot_pc2_genes, ["RETN", "GZMH", "GZMB"]]
    
    
    for i in range(5):
        print(f"PC genes in X.var_names: { [gene for gene in genes[i] if gene in X.var_names]}")
    
    
    # # pc1_comparison = compare_genes_with_pf2(X, top_pc1_genes, bot_pc1_genes, "PC1")
    # # pc2_comparison = compare_genes_with_pf2(X, top_pc2_genes, bot_pc2_genes, "PC2")
    # # print(pc1_comparison)
    # # print(pc2_comparison)
    
    # for i in range(len(top_pc1_genes)):
    #     plot_gene_pacmap(top_pc1_genes[i], "Pf2", X, ax[i])
        
    # cmp = 4
    # plot_gene_factors_partial(cmp, X, ax[5], top=True)
    
  

    return f


def plot_loadings_pca_partial(ax, geneAmount: int = 50, top=True, PC: int = 1):
    """XXX"""
    if PC == 1:
        df = pd.read_csv("loadings_time_series_PC1.csv", dtype=str).rename(columns={"Unnamed: 0": "Gene"}) 
    else:
        df = pd.read_csv("loadings_time_series_PC2.csv", dtype=str).rename(columns={"Unnamed: 0": "Gene"})
        
    df["PC"+str(PC)] = stats.zscore(df["PC"+str(PC)].to_numpy().astype(float))
    df = df.sort_values(by="PC"+str(PC))
    
    print(df)
    
    if top:
        sns.barplot(
            data=df.iloc[-geneAmount:, :], x="Gene", y="PC"+str(PC), color="k", ax=ax
            
        )
        higly_weighted_genes = df.iloc[-geneAmount:]["Gene"].values
    else:
        sns.barplot(data=df.iloc[:geneAmount, :], x="Gene", y="PC"+str(PC), color="k", ax=ax)
        higly_weighted_genes = df.iloc[:geneAmount]["Gene"].values

    ax.tick_params(axis="x", rotation=90)
    
    return df["Gene"].values    


# def compare_genes_with_pf2(X, top_pos_genes, top_neg_genes, pc_component, geneAmount: int = 20):
#     """Compare the top PCA genes with the genes in X.varm['Pf2_C'] and return a DataFrame."""
#     pf2_components = X.varm['Pf2_C']
#     overlap_counts = []

#     for i in range(pf2_components.shape[1]):
#         component_gene_names = X.var_names
#         pos_overlap = len(set(top_pos_genes).intersection(set(component_gene_names)))
#         neg_overlap = len(set(top_neg_genes).intersection(set(component_gene_names)))
#         total_overlap = pos_overlap + neg_overlap
#         overlap_counts.append({'PC Component': pc_component, 'PF2 Component': i+1, 'Overlap': total_overlap})

#     overlap_df = pd.DataFrame(overlap_counts)
#     overlap_df = overlap_df.sort_values(by='Overlap', ascending=False)
#     return overlap_df
    
# def plot_gene_factors_partial(
#     cmp: int, X, ax, geneAmount: int = 15, top=True
# ):
#     """Plotting weights for gene factors for both most negatively/positively weighted terms"""
#     cmpName = f"Cmp. {cmp}"

#     df = pd.DataFrame(
#         data=X.varm["Pf2_C"][:, cmp - 1], index=X.var_names, columns=[cmpName]
#     )
#     df = df.reset_index(names="Gene")
#     df = df.sort_values(by=cmpName)

#     if top:
#         sns.barplot(
#             data=df.iloc[-geneAmount:, :], x="Gene", color="k", ax=ax
#         )
#     else:
#         sns.barplot(data=df.iloc[:geneAmount, :], x="Gene", color="k", ax=ax)

#     ax.tick_params(axis="x", rotation=90)