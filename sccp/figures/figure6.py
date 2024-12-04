"""
XXX
"""

import anndata
import numpy as np
import pacmap
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
from .common import getSetup, subplotLabel
import seaborn as sns
import pandas as pd

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import stats
import mygene
import matplotlib.pyplot as plt

from .common import getSetup, subplotLabel
from .commonFuncs.plotFactors import plot_gene_factors
from .commonFuncs.plotGeneral import cell_count_perc_df, rotate_xaxis, avegene_per_status
from .commonFuncs.plotPaCMAP import plot_gene_pacmap, plot_wp_pacmap


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((18, 8), (4, 10))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/pf2/thomson_fitted.h5ad")
    n_components = 20
    
    pc = PCA(n_components=n_components)
    pca = pc.fit(np.asarray(X.X - X.var["means"].values))
    
    geneAmount = 30
    pc_load = load_pc_loadings(pca, X)
    pf2_factors = load_pf2_loadings(X)
    
    print(pc_load)
    print(pf2_factors)
    
    
    
    df = compare_pc_pf2_loadings_all_components(pc_load, pf2_factors, top_n=geneAmount)
    print(df.sort_values(by='Overlap_Count', ascending=False))
    
    # df1 = compare_pc_pf2_loadings(pc1_load, pf2_factors, pos_pca=True, pos_pf2=True, pc_component=1, top_n=geneAmount)
    # df2 = compare_pc_pf2_loadings(pc1_load, pf2_factors, pos_pca=True, pos_pf2=False, pc_component=1, top_n=geneAmount)
    # df3 = compare_pc_pf2_loadings(pc2_load, pf2_factors, pos_pca=False, pos_pf2=True, pc_component=2, top_n=geneAmount)
    # df4 = compare_pc_pf2_loadings(pc2_load, pf2_factors, pos_pca=False, pos_pf2=False, pc_component=2, top_n=geneAmount)

    # df = pd.concat([df1, df2, df3, df4], axis=0).reset_index()
    # print(df)
    
    # # df = df.sort_values(by='Overlap_Count', ascending=False).head(10)
    # # print(df)
    
    # overlapping_genes = df["Overlapping_Genes"].values
    # flattened_genes = [gene for sublist in overlapping_genes for gene in sublist]
    # genes = np.unique(flattened_genes)
    
    
    # for i, gene in enumerate(genes):
    #     df = avegene_per_status(X, gene=gene)
    #     # sns.boxplot(data=df, x="Status", y="Average Gene Expression", hue="Status", 
    #     #             showfliers=False, ax=ax[i])
    #     sns.boxplot(data=df, x="Cell Type", y="Average Gene Expression", hue="Status", 
    #                 showfliers=False, ax=ax[i])
        
    #     ax[i].tick_params(axis="x", rotation=45)
    #     ax[i].set_title(gene)


    # gene_corr_matrix = np.zeros((n_components, n_components))
    # for i in range(n_components):
    #     for j in range(n_components):
    #         corr, _ = spearmanr(pca.components_.T[:, i], X.varm["Pf2_C"][:, j])
    #         gene_corr_matrix[i, j] = corr
            
    # mask = np.triu(np.ones_like(gene_corr_matrix, dtype=bool))

    # for i in range(len(mask)):
    #     mask[i, i] = False
    # ticks = np.arange(1, gene_corr_matrix.shape[1]+1)
    # print(ticks)
    # sns.heatmap(
    #     data=gene_corr_matrix,
    #     # vmin=0.5,
    #     # vmax=1,
    #     center=0,
    #     cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True),
    #     xticklabels=ticks,
    #     yticklabels=ticks,
    #     mask=mask,
    #     cbar_kws={"label": "Prediction Accuracy"},
    #     ax=ax[0])
    
    
    
    for i in range(1, n_components):
        plot_gene_factors_partial(i, X, pca, ax[(2*i)-2], geneAmount=10, top=True)
        plot_gene_factors_partial(i, X, pca, ax[(2*i)+1-2], geneAmount=10, top=False)
        
        
    return f


def plot_gene_factors_partial(
    cmp: int, X, pca, ax, geneAmount: int = 5, top=True
):
    """Plotting weights for gene factors for both most negatively/positively weighted terms"""
    cmpName = f"Cmp. {cmp}"

    df = pd.DataFrame(
        data=pca.components_.T[:, cmp - 1], index=X.var_names, columns=[cmpName]
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
    
    


def load_pc_loadings(pca, X) -> pd.DataFrame:
    """
    Load and preprocess PC loadings for a specific component.

    Args:
        pc_component (int): Principal Component number (1 or 2)
        geneAmount (int): Number of genes to consider

    Returns:
        pd.DataFrame: Processed PC loadings DataFrame
    """
    df = pd.DataFrame(data=pca.components_.T, columns=[f"PC_{i}" for i in range(1, pca.components_.T.shape[1]+1)])
    df =  df.set_index(X.var_names).reset_index().rename(columns={"index": "Gene"})
    return df

def load_pf2_loadings(X) -> pd.DataFrame:
    """
    Load and preprocess PC loadings for a specific component.

    Args:
        pc_component (int): Principal Component number (1 or 2)
        geneAmount (int): Number of genes to consider

    Returns:
        pd.DataFrame: Processed PC loadings DataFrame
    """

    df = pd.DataFrame(data=X.varm['Pf2_C'], columns=[f"Pf2_{i}" for i in range(1, X.varm['Pf2_C'].shape[1]+1)])
    df =  df.set_index(X.var_names).reset_index().rename(columns={"index": "Gene"})

    return df
    



def compare_pc_pf2_loadings_all_components(pc_df: pd.DataFrame, pf2_df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """
    Compare top N positive loadings between all PC and Pf2 components.
    
    Args:
        pc_df (pd.DataFrame): DataFrame with PC loadings
        pf2_df (pd.DataFrame): DataFrame with Pf2 loadings
        top_n (int): Number of top genes to consider (default 30)
    
    Returns:
        pd.DataFrame: Comparison of top N positive loadings across all components
    """
    # Get number of PC and Pf2 components
    pc_components = [col for col in pc_df.columns if col.startswith('PC_')]
    pf2_components = [col for col in pf2_df.columns if col.startswith('Pf2_')]
    
    # Prepare results list to store comparisons
    results = []
    
    # Iterate through PC components
    for pc_component in pc_components:
        # Iterate through sign (positive and negative)
        for pos_pca in [True, False]:
            # Select PC loadings based on sign
            if pos_pca:
                pc_positive = pc_df[pc_df[pc_component] > 0].sort_values(pc_component, ascending=False).reset_index().head(top_n)
                pos_pca_name = "Pos"
            else:
                pc_positive = pc_df[pc_df[pc_component] < 0].sort_values(pc_component, ascending=True).reset_index().head(top_n)
                pos_pca_name = "Neg"
        
            # Iterate through Pf2 components
            for pf2_component in pf2_components:
                # Iterate through sign (positive and negative)
                for pos_pf2 in [True, False]:
                    # Select Pf2 loadings based on sign
                    if pos_pf2:
                        pf2_positive = pf2_df[pf2_df[pf2_component] > 0].sort_values(pf2_component, ascending=False).reset_index().head(top_n)
                        pos_pf2_name = "Pos"
                    else:
                        pf2_positive = pf2_df[pf2_df[pf2_component] < 0].sort_values(pf2_component, ascending=True).reset_index().head(top_n)
                        pos_pf2_name = "Neg"
                    
                    # Find overlapping genes in top N
                    overlap_genes = list(set(pc_positive['Gene']) & set(pf2_positive['Gene']))
                    overlap_count = len(overlap_genes)
                    
                    # Skip if no overlap
                    if overlap_count == 0:
                        continue
                    
                    # Get PC rankings 
                    if pos_pca:
                        pc_all_positive = pc_df[pc_df[pc_component] > 0].sort_values(pc_component, ascending=False).reset_index()
                    else:
                        pc_all_positive = pc_df[pc_df[pc_component] < 0].sort_values(pc_component, ascending=True).reset_index()
                    pc_rankings = [pc_all_positive[pc_all_positive['Gene'] == gene].index[0] + 1 for gene in overlap_genes]
                    
                    # Get Pf2 rankings
                    if pos_pf2:
                        pf2_all_positive = pf2_df[pf2_df[pf2_component] > 0].sort_values(pf2_component, ascending=False).reset_index()
                    else:
                        pf2_all_positive = pf2_df[pf2_df[pf2_component] < 0].sort_values(pf2_component, ascending=True).reset_index()
                    pf2_rankings = [pf2_all_positive[pf2_all_positive['Gene'] == gene].index[0] + 1 for gene in overlap_genes]
                    
                    # Prepare result row
                    result = {
                        'PC_Component': pc_component,
                        'Pf2_Component': pf2_component,
                        'Overlap_Count': overlap_count,
                        'Overlapping_Genes': overlap_genes,
                        'PC_Gene_Rankings': pc_rankings,
                        'Pf2_Gene_Rankings': pf2_rankings,
                        'PC_Sign': pos_pca_name,
                        'Pf2_Sign': pos_pf2_name
                    }
                    
                    results.append(result)
    
    # Convert to DataFrame
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame()  # Return empty DataFrame if no overlaps