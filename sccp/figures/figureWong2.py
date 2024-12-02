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
from .commonFuncs.plotGeneral import cell_count_perc_df, rotate_xaxis, avegene_per_status
from .commonFuncs.plotPaCMAP import plot_gene_pacmap, plot_wp_pacmap
from .figure4e_k import plot_correlation_cmp_cell_count_perc
from scipy import stats


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 8), (4, 4))
    subplotLabel(ax)
    
    
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    genes = ["FPR1", "PDE4B", "CLEC4E", "NFKBIZ", "IL1B", "CCL5", "SOCS3", "ARL4C", 
             "RGS1", "KLF4", "ISG15", "MX1", "TNFSF10", "ITGAL"]
    for i, gene in enumerate(genes):
        df = avegene_per_status(X, gene=gene)
        # sns.boxplot(data=df, x="Status", y="Average Gene Expression", hue="Status", 
        #             showfliers=False, ax=ax[i])
        sns.boxplot(data=df, x="Cell Type", y="Average Gene Expression", hue="Status", 
                    showfliers=False, ax=ax[i])
        
        ax[i].tick_params(axis="x", rotation=45)

    
   

    return f

