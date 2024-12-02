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
    top_pc1_genes = plot_loadings_pca_partial(ax[0], top=True, PC=1, geneAmount=geneAmount)
    bot_pc1_genes = plot_loadings_pca_partial(ax[1], top=False, PC=1, geneAmount=geneAmount)
    
    top_pc2_genes = plot_loadings_pca_partial(ax[2], top=True, PC=2, geneAmount=geneAmount)
    bot_pc2_genes = plot_loadings_pca_partial(ax[3], top=False, PC=2, geneAmount=geneAmount)

    genes = [top_pc1_genes, bot_pc1_genes, top_pc2_genes, bot_pc2_genes]
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
    
    return [gene.upper() for gene in higly_weighted_genes]

def compare_genes_with_pf2(X, top_pos_genes, top_neg_genes, pc_component, geneAmount: int = 40):
    """Compare the top PCA genes with the genes in X.varm['Pf2_C'] and return a DataFrame."""
    pf2_components = X.varm['Pf2_C']
    overlap_counts = []
    overlap_genes = []

    for i in range(pf2_components.shape[1]):
        component_genes_top = np.argsort(pf2_components[:, i])[-geneAmount:]
        component_genes_bottom = np.argsort(pf2_components[:, i])[:geneAmount]
        component_gene_names_top = X.var_names[component_genes_top]
        component_gene_names_bottom = X.var_names[component_genes_bottom]

        
        pos_overlap_top = set(top_pos_genes).intersection(set(component_gene_names_top))
        neg_overlap_top = set(top_neg_genes).intersection(set(component_gene_names_top))
        pos_overlap_bottom = set(top_pos_genes).intersection(set(component_gene_names_bottom))
        neg_overlap_bottom = set(top_neg_genes).intersection(set(component_gene_names_bottom))
        

        overlap_counts.append({
            'PC Component': pc_component,
            'PC Value': 'Pos',
            'PF2 Component': i+1,
            'PF2 Value': 'Pos',
            'Overlap': len(pos_overlap_top),
            'Overlapping Genes': ', '.join(pos_overlap_top)
        })
        overlap_counts.append({
            'PC Component': pc_component,
            'PC Value': 'Pos',
            'PF2 Component': i+1,
            'PF2 Value': 'Neg',
            'Overlap': len(pos_overlap_bottom),
            'Overlapping Genes': ', '.join(pos_overlap_bottom)
        })
        overlap_counts.append({
            'PC Component': pc_component,
            'PC Value': 'Neg',
            'PF2 Component': i+1,
            'PF2 Value': 'Pos',
            'Overlap': len(neg_overlap_top),
            'Overlapping Genes': ', '.join(neg_overlap_top)
        })
        overlap_counts.append({
            'PC Component': pc_component,
            'PC Value': 'Neg',
            'PF2 Component': i+1,
            'PF2 Value': 'Neg',
            'Overlap': len(neg_overlap_bottom),
            'Overlapping Genes': ', '.join(neg_overlap_bottom)
        })    


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