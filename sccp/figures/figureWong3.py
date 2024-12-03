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
        df_sorted = df.sort_values(by=f"PC{pc_component}")
        
        # Select top and bottom genes
        top_pos_genes = df_sorted.tail(geneAmount)
        top_neg_genes = df_sorted.head(geneAmount)
        
        return top_pos_genes, top_neg_genes
    except FileNotFoundError:
        raise ValueError(f"Loadings file for PC{pc_component} not found.")

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

def compare_genes_with_pf2(X, pc_component: int, geneAmount: int = 40, species='mouse'):
    """
    Compare top PCA genes with genes in X.varm['Pf2_C'].

    Args:
        X (AnnData): Annotated data matrix
        pc_component (int): Principal Component number
        geneAmount (int): Number of top/bottom genes to consider
        species (str): Species for gene conversion

    Returns:
        pd.DataFrame: Gene overlap and ranking analysis
    """
    # Load PC loadings
    top_pos_genes, top_neg_genes = load_pc_loadings(pc_component, geneAmount)
    
    # Convert gene symbols
    pc_pos_genes, _ = convert_gene_symbols(top_pos_genes['Gene'].tolist(), species)
    pc_neg_genes, _ = convert_gene_symbols(top_neg_genes['Gene'].tolist(), species)
    
    # Get Pf2 components
    pf2_components = X.varm['Pf2_C']
    
    # Prepare results dataframe
    results = []
    
    # Iterate through Pf2 components
    for pf2_idx, pf2_component in enumerate(pf2_components.T):
        # Get top and bottom genes for this Pf2 component
        pf2_sorted_indices = np.argsort(pf2_component)
        top_pf2_pos_genes = X.var_names[pf2_sorted_indices[-geneAmount:]].tolist()
        top_pf2_neg_genes = X.var_names[pf2_sorted_indices[:geneAmount]].tolist()
        
        # Convert Pf2 gene symbols
        pf2_pos_genes = top_pf2_pos_genes
        pf2_neg_genes = top_pf2_neg_genes
        
        # Compare positive genes
        pos_overlap = set(pc_pos_genes).intersection(set(pf2_pos_genes))
        pos_overlap_genes = list(pos_overlap)
        pos_overlap_count = len(pos_overlap_genes)
        
        # Compare negative genes
        neg_overlap = set(pc_neg_genes).intersection(set(pf2_neg_genes))
        neg_overlap_genes = list(neg_overlap)
        neg_overlap_count = len(neg_overlap_genes)
        
        # Prepare detailed results with gene rankings
        for overlap_type, pc_genes, pf2_genes, overlap_genes, overlap_count in [
            ('Positive', pc_pos_genes, pf2_pos_genes, pos_overlap_genes, pos_overlap_count),
            ('Negative', pc_neg_genes, pf2_neg_genes, neg_overlap_genes, neg_overlap_count)
        ]:
            # Prepare gene rankings
            pc_rankings = [pc_genes.index(gene) + 1 for gene in overlap_genes]
            pf2_rankings = [pf2_genes.index(gene) + 1 for gene in overlap_genes]
            
            results.append({
                'PC_Component': pc_component,
                'PC_Sign': overlap_type,
                'Pf2_Component': pf2_idx+1,
                'Pf2_Sign': overlap_type,
                'Overlap_Count': overlap_count,
                'Overlapping_Genes': overlap_genes,
                'PC_Gene_Rankings': pc_rankings,
                'Pf2_Gene_Rankings': pf2_rankings
            })
    
    return pd.DataFrame(results)




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
    geneAmount = 40

    # Load data
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    df_pc1 = compare_genes_with_pf2(X, 1, geneAmount=geneAmount, species='mouse')
    df_pc2 = compare_genes_with_pf2(X, 2, geneAmount=geneAmount, species='mouse')
    df = pd.concat([df_pc1, df_pc2], axis=0)
    df = df.sort_values(by='Overlap_Count', ascending=False)

    print(df.iloc[:10])
    
    
    return f