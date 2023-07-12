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
# lupus_data = anndata.read_h5ad("/home/seanp/scCP/GSE174188_CLUES1_adjusted.h5ad")
lupus_data = anndata.read_h5ad("/opt/andrew/lupus/lupus.h5ad")

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

# genes with highest "fold change":
# I have become increasingly skeptical about the validity of this
# Found using: print(lupus_genes.sort_values("fold_change", ascending=False))

# DERL3          ENSG00000099958   128.664734
# MIB2           ENSG00000197530    61.606426
# POU2AF1        ENSG00000110777    31.871082
# ZNF595         ENSG00000197701    26.963999
# LINC00685      ENSG00000226179    20.619701

# make .rnk file to use with WebGestalt
#lupus_genes.to_csv('fold_change.rnk', index=False, header=False, sep='\t')


############################################################################################
# SEAN MAKES SOME SCATTERPLOTS ########
############################################################################################

# resetting the combo_lupus association because here I want all observational things to map more variables to color etc

lupus_pan = lupus_data.to_df()

lupus_observations = lupus_data.obs

combo_lupus = lupus_observations.merge(lupus_pan, 
                                       how= "left",
                                       left_index=True,
                                       right_index=True,
                                       validate="one_to_one")


# get average by donor_id

mean_gene_dictionary = {}
# add all the genes, and tell it to calc mean for each
for gene in lupus_genes.index.tolist():
    mean_gene_dictionary[gene] = "mean"

avg_combo_lupus = combo_lupus.groupby(["ind_cov"]).agg(mean_gene_dictionary)

# want to combine with col abt pop_cov and SLE status
obs_cov_status = lupus_observations[["ind_cov", "SLE_status", "pop_cov"]].drop_duplicates(subset="ind_cov")


lupus_by_patient = pd.merge(obs_cov_status, avg_combo_lupus, right_index=True, left_on="ind_cov")


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), # fig size
                     (3, 2) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    print(lupus_by_patient)

    # added CCR7 to look at T cell differences
    high_de_genes = ["DERL3", "MIB2", "POU2AF1", "ZNF595", "LINC00685", "CCR7"]  
    counter = 0

    for gene in high_de_genes:
        sns.violinplot(data= lupus_by_patient,  y = gene, x = "SLE_status", hue = "pop_cov", dodge=True, linewidth=0.1, ax=ax[counter])
        sns.stripplot(data = lupus_by_patient, y = gene, x = "SLE_status", hue = "pop_cov", size=2, dodge=True, legend=False, ax=ax[counter])
        ax[counter].set_title(gene + " Expression")
        ax[counter].legend_.remove()
        counter += 1

    # got this list from seaching "T cell markers" on google. This output shows only CCR7 is in the dataset (at least w same name)
    possible_t_cell_markers = ["CD45RA", "CD62L", "CD127", "CD132", "CCR7"]

    for marker in possible_t_cell_markers:
        print(marker, ":", marker in lupus_genes.index.tolist())

    return f