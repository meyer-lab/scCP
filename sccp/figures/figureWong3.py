"""
Figure 5a_e Generation Script

This module generates a comprehensive figure comparing PCA and PF2 component analyses 
for gene loadings and factors.
"""

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
from .commonFuncs.plotGeneral import cell_count_perc_df, rotate_xaxis
from .commonFuncs.plotPaCMAP import plot_gene_pacmap, plot_wp_pacmap


def load_pc_loadings(pc_component: int, geneAmount: int = 40) -> pd.DataFrame:
    """
    Load and preprocess PC loadings for a specific component.

    Args:
        pc_component (int): Principal Component number (1 or 2)
        geneAmount (int): Number of genes to consider

    Returns:
        pd.DataFrame: Processed PC loadings DataFrame
    """
    try:
        df = pd.read_csv(f"loadings_time_series_PC{pc_component}.csv", dtype=str)
        df = df.rename(columns={"Unnamed: 0": "Gene"})
        df[f"PC{pc_component}"] = stats.zscore(df[f"PC{pc_component}"].astype(float))
        return df.sort_values(by=f"PC{pc_component}")
    except FileNotFoundError:
        raise ValueError(f"Loadings file for PC{pc_component} not found.")


def plot_loadings_pca(
    ax, 
    pc_component: int, 
    geneAmount: int = 40, 
    top: bool = True
):
    """
    Plot PCA loadings for a specific component.

    Args:
        ax (plt.Axes): Matplotlib axes to plot on
        pc_component (int): Principal Component number
        geneAmount (int): Number of top/bottom genes to plot
        top (bool): Whether to plot top or bottom genes

    Returns:
        list: List of top/bottom gene names
    """
    df = load_pc_loadings(pc_component, geneAmount)
    
    if top:
        plot_df = df.iloc[-geneAmount:, :]
        highly_weighted_genes = df.iloc[-geneAmount:]["Gene"].values
    else:
        plot_df = df.iloc[:geneAmount, :]
        highly_weighted_genes = df.iloc[:geneAmount]["Gene"].values

    sns.barplot(
        data=plot_df, 
        x="Gene", 
        y=f"PC{pc_component}", 
        color="k", 
        ax=ax
    )
    ax.tick_params(axis="x", rotation=90)
    
    return [gene.upper() for gene in highly_weighted_genes]


def convert_gene_symbols(genes, species='mouse'):
    """
    Convert gene symbols using MyGene.

    Args:
        genes (list): List of gene symbols
        species (str): Species for gene conversion

    Returns:
        tuple: Converted genes and list of genes without hits
    """
    mg = mygene.MyGeneInfo()
    results = mg.querymany(
        genes, 
        scopes='symbol', 
        fields=['symbol', 'entrezgene'], 
        species=species, 
        transformed=True
    )

    conversion_map = []
    no_hit_genes = []
    
    for gene, result in zip(genes, results):
        if result and 'symbol' in result:
            conversion_map.append(result.get('symbol', gene).upper())
        else:
            conversion_map.append(gene.upper())
            no_hit_genes.append(gene)
    
    return conversion_map, no_hit_genes


def compare_genes_with_pf2(
    X, 
    pc_pos_genes, 
    pc_neg_genes, 
    pc_component, 
    geneAmount: int = 40
):
    """
    Compare top PCA genes with genes in X.varm['Pf2_C'].

    Args:
        X (AnnData): Annotated data matrix
        pc_pos_genes (list): Top positive genes for a specific PC
        pc_neg_genes (list): Top negative genes for a specific PC
        pc_component (int): Principal Component number
        geneAmount (int): Number of top/bottom genes to consider

    Returns:
        pd.DataFrame: Gene overlap and ranking analysis
    """
    pf2_components = X.varm['Pf2_C']
    overlap_counts = []

    for i in range(pf2_components.shape[1]):
        # Get top and bottom genes for the current PF2 component
        pf2_pos_genes_indices = np.argsort(pf2_components[:, i])[-geneAmount:]
        pf2_neg_genes_indices = np.argsort(pf2_components[:, i])[:geneAmount]
        
        pf2_pos_genes = X.var_names[pf2_pos_genes_indices]
        pf2_neg_genes = X.var_names[pf2_neg_genes_indices]

        def get_pc_gene_ranking(pc_pos_genes, gene):
            """
            Calculate the ranking of a gene within a list of positively ranked genes.

            Args:
                pc_pos_genes (np.array): Array of genes sorted by their loading/importance
                gene (str): Gene to find the ranking for

            Returns:
                int: Ranking of the gene (from the end of the array), or 0 if not found
            """
            try:
                # Find indices where the gene matches in the array
                gene_indices = np.where(pc_pos_genes == gene)[0]
                
                # If the gene is found, return its ranking
                if len(gene_indices) > 0:
                    return len(pc_pos_genes) - gene_indices[0]
                
                # If gene not found, return 0 or another appropriate default
                return 0
            
            except Exception as e:
                print(f"Error in gene ranking for {gene}: {e}")
            return 0

        def get_pf2_gene_ranking(gene, pf2_pos_genes, pf2_neg_genes):
            """Calculate PF2 gene ranking."""
            try:
                if gene in pf2_pos_genes:
                    return len(pf2_pos_genes) - np.where(pf2_pos_genes == gene)[0][0]
                if gene in pf2_neg_genes:
                    return len(pf2_neg_genes) - np.where(pf2_neg_genes == gene)[0][0]
                return None
            except IndexError:
                return None

        # Find overlaps between PC and PF2 gene sets
        overlap_scenarios = [
            ('Positive', 'Positive', set(pc_pos_genes).intersection(set(pf2_pos_genes))),
            ('Positive', 'Negative', set(pc_pos_genes).intersection(set(pf2_neg_genes))),
            ('Negative', 'Positive', set(pc_neg_genes).intersection(set(pf2_pos_genes))),
            ('Negative', 'Negative', set(pc_neg_genes).intersection(set(pf2_neg_genes)))
        ]

        for pc_category, pf2_category, overlap_genes in overlap_scenarios:
            pc_gene_rankings = []
            pf2_gene_rankings = []
            
            for gene in overlap_genes:
                pc_ranking = get_pc_gene_ranking(
                    gene, 
                    pc_pos_genes if pc_category == 'Positive' else pc_neg_genes, 
                    pc_neg_genes if pc_category == 'Positive' else pc_pos_genes
                )
                pf2_ranking = get_pf2_gene_ranking(
                    gene, 
                    pf2_pos_genes if pf2_category == 'Positive' else pf2_neg_genes, 
                    pf2_neg_genes if pf2_category == 'Positive' else pf2_pos_genes
                )
                
                if pc_ranking is not None:
                    pc_gene_rankings.append(str(pc_ranking))
                if pf2_ranking is not None:
                    pf2_gene_rankings.append(str(pf2_ranking))
            
            overlap_counts.append({
                'PC_Component': pc_component,
                'PC_Component_Category': pc_category,
                'PF2_Component': i+1,
                'PF2_Component_Category': pf2_category,
                'Overlap_Size': len(overlap_genes),
                'Overlapping_Genes': ', '.join(overlap_genes),
                'PC_Gene_Rankings': ', '.join(pc_gene_rankings) if pc_gene_rankings else '',
                'PF2_Gene_Rankings': ', '.join(pf2_gene_rankings) if pf2_gene_rankings else ''
            })

    return pd.DataFrame(overlap_counts)


def plot_gene_factors_partial(
    cmp: int, 
    X, 
    ax, 
    geneAmount: int = 20, 
    top: bool = True
):
    """
    Plot gene factor weights for a specific component.

    Args:
        cmp (int): Component number
        X (AnnData): Annotated data matrix
        ax (plt.Axes): Matplotlib axes to plot on
        geneAmount (int): Number of genes to plot
        top (bool): Whether to plot top or bottom genes
    """
    cmpName = f"Cmp. {cmp}"

    df = pd.DataFrame(
        data=X.varm["Pf2_C"][:, cmp - 1], 
        index=X.var_names, 
        columns=[cmpName]
    )
    df = df.reset_index(names="Gene")
    df = df.sort_values(by=cmpName)

    if top:
        sns.barplot(
            data=df.iloc[-geneAmount:, :], 
            x="Gene", 
            y=cmpName, 
            color="k", 
            ax=ax
        )
    else:
        sns.barplot(
            data=df.iloc[:geneAmount, :], 
            x="Gene", 
            y=cmpName, 
            color="k", 
            ax=ax
        )

    ax.tick_params(axis="x", rotation=90)


def makeFigure():
    """
    Generate the comprehensive figure comparing PCA and PF2 components.

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    # Figure setup
    ax, f = getSetup((12, 6), (4, 2))
    subplotLabel(ax)

    # Parameters
    geneAmount = 30

    # Plot PCA loadings for PC1 and PC2
    pos_pc1_genes = plot_loadings_pca(ax[0], pc_component=1, top=True, geneAmount=geneAmount)
    neg_pc1_genes = plot_loadings_pca(ax[1], pc_component=1, top=False, geneAmount=geneAmount)
    
    pos_pc2_genes = plot_loadings_pca(ax[2], pc_component=2, top=True, geneAmount=geneAmount)
    neg_pc2_genes = plot_loadings_pca(ax[3], pc_component=2, top=False, geneAmount=geneAmount)

    # Convert gene symbols
    genes = [pos_pc1_genes, neg_pc1_genes, pos_pc2_genes, neg_pc2_genes]
    converted_genes = [convert_gene_symbols(gene_list)[0] for gene_list in genes]

    # Load data
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    # Compare genes
    df_pc1 = compare_genes_with_pf2(X, converted_genes[0], converted_genes[1], 1, geneAmount=geneAmount)
    df_pc2 = compare_genes_with_pf2(X, converted_genes[2], converted_genes[3], 2, geneAmount=geneAmount)
    df = pd.concat([df_pc1, df_pc2], axis=0)
    
    # Print results
    print(df)
    print(df.iloc[:10])

    # Plot gene factors
    plot_gene_factors_partial(13, X, ax[4], top=False)
    plot_gene_factors_partial(24, X, ax[5], top=False)
    plot_gene_factors_partial(16, X, ax[6], top=False)
    plot_gene_factors_partial(11, X, ax[7], top=True)

    return f