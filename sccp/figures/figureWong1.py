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


    geneAmount = 30
    pos_pc1_genes, df_pc1 = plot_loadings_pca_partial(ax[0], top=True, PC=1, geneAmount=geneAmount)
    neg_pc1_genes, _ = plot_loadings_pca_partial(ax[1], top=False, PC=1, geneAmount=geneAmount)
    
    pos_pc2_genes, df_pc2 = plot_loadings_pca_partial(ax[2], top=True, PC=2, geneAmount=geneAmount)
    neg_pc2_genes, _ = plot_loadings_pca_partial(ax[3], top=False, PC=2, geneAmount=geneAmount)

    genes = [pos_pc1_genes, neg_pc1_genes, pos_pc2_genes, neg_pc2_genes]
    for i in range(len(genes)):
        mg = mygene.MyGeneInfo()
        results = mg.querymany(
            genes[i], 
            scopes='symbol', 
            fields=['symbol', 'entrezgene'], 
            species='mouse', 
            transformed=True
        )

        conversion_map = []
        no_hit_genes = []
    
        for gene, result in zip(genes[i], results):
            if result and 'symbol' in result:
                # Successful conversion
                conversion_map.append(result.get('symbol'))
            else:
                # No conversion found
                conversion_map.append(gene)
                no_hit_genes.append(gene)
    
        genes[i] = [gene.upper() for gene in conversion_map]
    
    
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    df_pc1 = compare_genes_with_pf2(X, genes[0], genes[1], 1, geneAmount=geneAmount)
    df_pc2 = compare_genes_with_pf2(X, genes[2], genes[3], 2, geneAmount=geneAmount)
    df = pd.concat([df_pc1, df_pc2], axis=0)
    
    df = df.sort_values(by='Overlap', ascending=False)

    print(df.iloc[:10])
    
    
    plot_gene_factors_partial(1, X, ax[4], top=True)
    plot_gene_factors_partial(1, X, ax[5], top=False)
    plot_gene_factors_partial(16, X, ax[6], top=True)
    plot_gene_factors_partial(16, X, ax[7], top=False)


    return f


def plot_loadings_pca_partial(ax, geneAmount: int = 40, top=True, PC: int = 1):
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

def compare_genes_with_pf2(X, pc_pos_genes, pc_neg_genes, pc_component, geneAmount: int = 40):
    """Compare the top PCA genes with the genes in X.varm['Pf2_C'] and return a DataFrame."""
    pf2_components = X.varm['Pf2_C']
    overlap_counts = []

    for i in range(pf2_components.shape[1]):
        # Get top and bottom genes for the current PF2 component
        pf2_pos_genes_indices = np.argsort(pf2_components[:, i])[-geneAmount:]
        pf2_neg_genes_indices = np.argsort(pf2_components[:, i])[:geneAmount]
        
        pf2_pos_genes = X.var_names[pf2_pos_genes_indices]
        pf2_neg_genes = X.var_names[pf2_neg_genes_indices]

        def get_gene_ranking(gene, pc_pos_genes, pc_neg_genes, pf2_pos_genes, pf2_neg_genes):
            """
            Calculate the ranking of a gene in different gene lists.
            
            Returns the ranking (1-based index) or None if not found.
            """
            try:
                if gene in pc_pos_genes:
                    return np.where(pc_pos_genes == gene)[0][0] + 1
                if gene in pc_neg_genes:
                    return np.where(pc_neg_genes == gene)[0][0] + 1
                if gene in pf2_pos_genes:
                    return np.where(pf2_pos_genes == gene)[0][0] + 1
                if gene in pf2_neg_genes:
                    return np.where(pf2_neg_genes == gene)[0][0] + 1
                return None
            except IndexError:
                return None

        # Find overlaps between PC and PF2 gene sets
        pc_pos_pf2_pos_overlap = set(pc_pos_genes).intersection(set(pf2_pos_genes))
        pc_pos_pf2_neg_overlap = set(pc_pos_genes).intersection(set(pf2_neg_genes))
        pc_neg_pf2_pos_overlap = set(pc_neg_genes).intersection(set(pf2_pos_genes))
        pc_neg_pf2_neg_overlap = set(pc_neg_genes).intersection(set(pf2_neg_genes))

        # Prepare overlap information with rankings
        def create_overlap_entry(pc_category, pf2_category, overlap_genes):
            return {
                'PC_Component': pc_component,
                'PC_Component_Category': pc_category,
                'PF2_Component': i+1,
                'PF2_Component_Category': pf2_category,
                'Overlap': len(overlap_genes),
                'Overlapping_Genes': ', '.join(overlap_genes),
                'PC_Gene_Rankings': ', '.join(
                    str(ranking) for ranking in 
                    (get_gene_ranking(gene, pc_pos_genes, pc_neg_genes, pf2_pos_genes, pf2_neg_genes) for gene in overlap_genes) 
                    if ranking is not None
                ),
                'PF2_Gene_Rankings': ', '.join(
                    str(ranking) for ranking in 
                    (get_gene_ranking(gene, pf2_pos_genes, pf2_neg_genes, pc_pos_genes, pc_neg_genes) for gene in overlap_genes) 
                    if ranking is not None
                )
            }

        # Create entries for different overlap scenarios
        overlap_counts.extend([
        create_overlap_entry('Positive', 'Positive', pc_pos_pf2_pos_overlap),
        create_overlap_entry('Positive', 'Negative', pc_pos_pf2_neg_overlap),
        create_overlap_entry('Negative', 'Positive', pc_neg_pf2_pos_overlap),
        create_overlap_entry('Negative', 'Negative', pc_neg_pf2_neg_overlap)
        ])

    # Convert to DataFrame
    overlap_df = pd.DataFrame(overlap_counts)
    
    return overlap_df

    
def plot_gene_factors_partial(
    cmp: int, X, ax, geneAmount: int = 20, top=True
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
    
    ret