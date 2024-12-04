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
from .figureWong4 import compare_pc_pf2_loadings, load_pc_loadings



def makeFigure():
    ax, f = getSetup((12, 8), (4, 4))
    subplotLabel(ax)
    
    geneAmount = 40
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    df_pc1 = load_pc_loadings(pc_component=1)
    # df_pc2 = load_pc_loadings(pc_component=2)
    # df = pd.concat([df_pc1, df_pc2], axis=0)
    # df = df.sort_values(by='Overlap_Count', ascending=False)
    
    
    overlap_genes = list(set(df_pc1['Gene']) & set(X.var_names))
    print(overlap_genes)
    overlap_count = len(overlap_genes)
    print(overlap_count)

    # df = df.iloc[:10]
    # overlapping_genes = df["Overlapping_Genes"].values
    # flattened_genes = [gene for sublist in overlapping_genes for gene in sublist]
    # genes = np.unique(flattened_genes)
    
    
    # for i, gene in enumerate(genes):
    #     df = avegene_per_status(X, gene=gene)
    #     sns.boxplot(data=df, x="Status", y="Average Gene Expression", hue="Status", 
    #                 showfliers=False, ax=ax[i])
    #     # sns.boxplot(data=df, x="Cell Type", y="Average Gene Expression", hue="Status", 
    #     #             showfliers=False, ax=ax[i])
        
    #     ax[i].tick_params(axis="x", rotation=45)

    
   

    return f

