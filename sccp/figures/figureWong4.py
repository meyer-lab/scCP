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
from .commonFuncs.plotGeneral import cell_count_perc_df, rotate_xaxis, avegene_per_status
from .commonFuncs.plotPaCMAP import plot_gene_pacmap, plot_wp_pacmap



def makeFigure():
    ax, f = getSetup((12, 8), (4, 5))
    subplotLabel(ax)
    
    geneAmount = 5
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    pc1_load = load_pc_loadings(pc_component=1)
    pc2_load = load_pc_loadings(pc_component=2)
    
    # pf2_factors = load_pf2_loadings(X)
    
    
    overlap_genes = list(set(pc1_load['Gene']) & set(X.var_names))
    print(len(overlap_genes))
    overlap_genes = list(set(pc2_load['Gene']) & set(X.var_names))
    print(len(overlap_genes))
    pc1_load = pc1_load[pc1_load['Gene'].isin(overlap_genes)].sort_values(by='PC1', ascending=False)
    genes = pc1_load['Gene'].values
    
    
    for i, gene in enumerate(np.concatenate([genes[:geneAmount], genes[-geneAmount:]])):
        df = avegene_per_status(X, gene=gene)
        sns.boxplot(data=df, x="Status", y="Average Gene Expression", hue="Status", 
                    showfliers=False, ax=ax[i])
        # sns.boxplot(data=df, x="Cell Type", y="Average Gene Expression", hue="Status", 
        #             showfliers=False, ax=ax[i])
        
        ax[i].tick_params(axis="x", rotation=45)
        if 0 <= i < 5:
            ax[i].set_title("PC1 Pos: "+ gene)
        if i >= 5:
            ax[i].set_title("PC1 Neg: "+ gene)
            

    pc2_load = pc2_load[pc2_load['Gene'].isin(overlap_genes)].sort_values(by='PC2', ascending=False)
    genes = pc2_load['Gene'].values
    
    
    for i, gene in enumerate(np.concatenate([genes[:geneAmount], genes[-geneAmount:]])):
        df = avegene_per_status(X, gene=gene)
        sns.boxplot(data=df, x="Status", y="Average Gene Expression", hue="Status", 
                    showfliers=False, ax=ax[i+10])
        # sns.boxplot(data=df, x="Cell Type", y="Average Gene Expression", hue="Status", 
        #             showfliers=False, ax=ax[i+10])
        
        ax[i+10].tick_params(axis="x", rotation=45)
        if 0 <= i < 5:
            ax[i+10].set_title("PC2 Pos: "+ gene)
        if i >= 5:
            ax[i+10].set_title("PC2 Neg: "+ gene)
            
            
            
            

    
       

    
    
    
    
   


    
    # df1 = compare_pc_pf2_loadings(pc1_load, pf2_factors, pos_pca=True, pos_pf2=True, pc_component=1, top_n=geneAmount)
    # df2 = compare_pc_pf2_loadings(pc1_load, pf2_factors, pos_pca=True, pos_pf2=False, pc_component=1, top_n=geneAmount)
    # df3 = compare_pc_pf2_loadings(pc2_load, pf2_factors, pos_pca=False, pos_pf2=True, pc_component=2, top_n=geneAmount)
    # df4 = compare_pc_pf2_loadings(pc2_load, pf2_factors, pos_pca=False, pos_pf2=False, pc_component=2, top_n=geneAmount)

    # df = pd.concat([df1, df2, df3, df4], axis=0).reset_index()
    # print(df)
    
    # df = df.sort_values(by='Overlap_Count', ascending=False).head(10)
    # print(df)
    
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

    
       


    return f


def load_pc_loadings(pc_component: int) -> pd.DataFrame:
    """
    Load and preprocess PC loadings for a specific component.

    Args:
        pc_component (int): Principal Component number (1 or 2)
        geneAmount (int): Number of genes to consider

    Returns:
        pd.DataFrame: Processed PC loadings DataFrame
    """

    df = pd.read_csv(f"loadings_time_series_PC{pc_component}.csv", dtype=str)
    df = df.rename(columns={"Unnamed: 0": "Gene"})
    df["Gene"] = convert_gene_symbols(df["Gene"])[0]
    df[f"PC{pc_component}"] = stats.zscore(df[f"PC{pc_component}"].astype(float))

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
    


def convert_gene_symbols(genes):
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
        species='mouse', 
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



def compare_pc_pf2_loadings(pc_df: pd.DataFrame, pf2_df: pd.DataFrame, top_n: int = 30, pos_pca = True, pos_pf2 = True, pc_component=1) -> pd.DataFrame:
    """
    Compare top N positive loadings between PC and Pf2 components.
    
    Args:
        pc_df (pd.DataFrame): DataFrame with PC loadings
        pf2_df (pd.DataFrame): DataFrame with Pf2 loadings
        top_n (int): Number of top genes to consider (default 30)
    
    Returns:
        pd.DataFrame: Comparison of top N positive loadings
    """
    # Prepare results list to store comparisons
    results = []
    if pos_pca is True:
        pos_pca_name = "Pos"
    elif pos_pca is False:
        pos_pca_name = "Neg"
    if pos_pf2 is True:
        pos_pf2_name = "Pos"
    elif pos_pf2 is False:
        pos_pf2_name = "Neg"
    
    # Get top N PC positive loadings
    if pos_pca:
        pc_positive = pc_df[pc_df[f'PC{pc_component}'] > 0].sort_values(f'PC{pc_component}', ascending=False).reset_index().head(top_n)
    else:
        pc_positive = pc_df[pc_df[f'PC{pc_component}'] < 0].sort_values(f'PC{pc_component}', ascending=True).reset_index().head(top_n)
        
    
    # Iterate through Pf2 components
    for pf2_idx, pf2_col in enumerate([col for col in pf2_df.columns if col.startswith('Pf2_')]):
        # Get top N positive Pf2 loadings 
        pf2_df[["Gene", pf2_col]]
        if pos_pf2:
            pf2_positive = pf2_df[pf2_df[pf2_col] > 0].sort_values(pf2_col, ascending=False).reset_index().head(top_n)
        else:
            pf2_positive = pf2_df[pf2_df[pf2_col] < 0].sort_values(pf2_col, ascending=True).reset_index().head(top_n)
        pf2_positive = pf2_positive[["Gene", pf2_col]]  
        
        # Find overlapping genes in top 30
        overlap_genes = list(set(pc_positive['Gene']) & set(pf2_positive['Gene']))
        overlap_count = len(overlap_genes)
        
        # Skip if no overlap
        if overlap_count == 0:
            continue
        
        # Get PC rankings (overall in positive values)
        if pos_pca:
            pc_all_positive = pc_df[pc_df[f'PC{pc_component}'] > 0].sort_values(f'PC{pc_component}', ascending=False)
        else:
            pc_all_positive = pc_df[pc_df[f'PC{pc_component}'] < 0].sort_values(f'PC{pc_component}', ascending=True).reset_index()
        pc_rankings = [pc_all_positive[pc_all_positive['Gene'] == gene].index[0] + 1 for gene in overlap_genes]
        
        # Get Pf2 rankings (within top 30)
        pf2_rankings = [pf2_positive[pf2_positive['Gene'] == gene].index[0] + 1 for gene in overlap_genes]
        
        # Prepare result row
        result = {
            'PC_Component': pc_component,
            'Pf2_Component': pf2_idx+1,
            'Overlap_Count': overlap_count,
            'Overlapping_Genes': overlap_genes,
            'PC_Gene_Rankings': pc_rankings,
            'Pf2_Gene_Rankings': pf2_rankings
        }
        
        results.append(result)
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        df["PC_Sign"] = pos_pca_name
        df["Pf2_Sign"] = pos_pf2_name
        return df
    else:
        return pd.DataFrame()  # Return empty DataFrame if no overlaps
