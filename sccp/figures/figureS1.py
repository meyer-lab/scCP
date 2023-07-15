"""
S1: Initial visualizations of Lupus Data Pt2: distributions of genes
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: investigate the normalization methods used on the dataset as published

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup
)
import pandas as pd
import seaborn as sns
import numpy as np
import anndata
import random


# load data (need to change filepath once dataset gets uploaded to opt/...)
# lupus_data = anndata.read_h5ad("/home/seanp/scCP/GSE174188_CLUES1_adjusted.h5ad")
lupus_data = anndata.read_h5ad("/opt/andrew/lupus/lupus.h5ad")


# get observational variables combined
lupus_pan = lupus_data.to_df()


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

    description = lupus_pan.describe()



    gene_list = lupus_genes.index.tolist()

    random.seed(1)
    rand_genes = random.choices(gene_list, k= 6)

    #sns.histplot(data = description.T, x = "max", kde=True)
    #ax[0].set_title("Max Gene Expression in Normalized Lupus Dataset")
    
    # can uncomment above and comment below to show the max gene expression
    # throughout. Otherwise this serves mostly to visualize the way
    # this dataset was normalized

    counter = 0

    for gene in rand_genes:
        ax[counter].set_xlim([-1, 1])
        sns.histplot(data= combo_lupus,  x = gene, ax=ax[counter], bins=300)
        ax[counter].set_title(gene + " Expression")
        counter += 1

    return f