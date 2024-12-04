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
from .figureWong4 import compare_pc_pf2_loadings, load_pc_loadings, load_pf2_loadings


def makeFigure():
    ax, f = getSetup((5, 5), (1, 1))
    subplotLabel(ax)
    
    geneAmount = 35
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
      
    pc1_load = load_pc_loadings(pc_component=1)
    pc2_load = load_pc_loadings(pc_component=2)
    
    pf2_factors = load_pf2_loadings(X)
    
    df1 = compare_pc_pf2_loadings(pc1_load, pf2_factors, pos_pca=True, pos_pf2=True, pc_component=1, top_n=geneAmount)
    df2 = compare_pc_pf2_loadings(pc1_load, pf2_factors, pos_pca=True, pos_pf2=False, pc_component=1, top_n=geneAmount)
    df3 = compare_pc_pf2_loadings(pc2_load, pf2_factors, pos_pca=False, pos_pf2=True, pc_component=2, top_n=geneAmount)
    df4 = compare_pc_pf2_loadings(pc2_load, pf2_factors, pos_pca=False, pos_pf2=False, pc_component=2, top_n=geneAmount)

    df = pd.concat([df1, df2, df3, df4], axis=0).reset_index()
    
    sns.scatterplot(data=df, x="PC_Gene_Rankings", y="Pf2_Gene_Rankings", hue="PC_Sign", style="Pf2_Sign", ax=ax[0])

    
    
    
    return f

