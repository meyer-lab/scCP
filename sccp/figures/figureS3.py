"""
S3: Initial visualizations of Lupus Data Pt2: distributions of genes
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
import random


# load data (need to change filepath once dataset gets uploaded to opt/...)
# lupus_data = anndata.read_h5ad("/home/seanp/scCP/GSE174188_CLUES1_adjusted.h5ad")
lupus_data = anndata.read_h5ad("/opt/andrew/lupus/lupus.h5ad")

# get rid of warnings <3 (this lowkey doesn't work)
warnings.filterwarnings("once")


# get observational variables combined
lupus_pan = lupus_data.to_df()

lupus_pan_center = lupus_data.to_df().apply(lambda x: x-x.mean())



lupus_observations = lupus_data.obs[["SLE_status"]]

combo_lupus = lupus_observations.merge(lupus_pan, 
                                       how= "left",
                                       left_index=True,
                                       right_index=True,
                                       validate="one_to_one")


lupus_genes = lupus_data.var



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), # fig size
                     (3, 2) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)
    # funky. it mostly is centered already, also some genes don't go up to 10. idk what's up
    print(lupus_pan.describe())
    print(lupus_pan_center.describe())


    # added CCR7 to look at T cell differences
    gene_list = lupus_genes.index.tolist()

    random.seed(1)
    rand_genes = random.choices(gene_list, k= 5)

    counter = 0

    for gene in rand_genes:
        ax[counter].set_xlim([-1, 1])
        sns.histplot(data= combo_lupus,  x = gene, ax=ax[counter], bins=300)
        ax[counter].set_title(gene + " Expression")
        counter += 1

    return f
