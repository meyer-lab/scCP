"""
S2: GSEA + Initial visualizations of Lupus Data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup,
    flattenData
)
from ..imports.scRNA import import_pancreas, import_pancreas_all
from ..parafac2 import parafac2_nd
import pandas as pd
import seaborn as sns
from os.path import dirname
import numpy as np
from scipy.stats import linregress
import anndata
import scipy 
import warnings


# load data (need to change filepath once dataset gets uploaded to opt/...)
lupus_data = anndata.read_h5ad("/home/seanp/scCP/GSE174188_CLUES1_adjusted.h5ad")

# get rid of warnings <3 (this lowkey doesn't work)
warnings.filterwarnings("once")


# get observational variables combined
lupus_pan = lupus_data.to_df()

lupus_observations = lupus_data.obs[["SLE_status"]]

combo_lupus = lupus_observations.merge(lupus_pan, 
                                       how= "left",
                                       left_index=True,
                                       right_index=True,
                                       validate="one_to_one")


############################################################################################
# SEAN MAKES A GSEA ATTEMPT ########
############################################################################################


healthy_vals = combo_lupus[combo_lupus["SLE_status"] == "Healthy"].iloc[:, 1:].values
SLE_vals = combo_lupus[combo_lupus["SLE_status"] == "SLE"].iloc[:, 1:].values

lupus_genes = lupus_data.var

lupus_genes['fold_change'] = np.mean(SLE_vals, axis=0)/np.mean(healthy_vals, axis=0)

#lupus_genes.to_csv('fold_change.rnk', index=False, header=False, sep='\t')



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), # fig size
                     (1, 1) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    #print(combo_lupus)
    #print(healthy_vals)
    print(lupus_genes)
 





    return f



