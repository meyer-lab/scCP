"""
loading in + playing with lupus data
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
import warnings


# load data 
lupus_data = anndata.read_h5ad("/opt/andrew/lupus/lupus.h5ad")

# get rid of warnings <3 (this lowkey doesn't work)
warnings.filterwarnings("once")

# make a pandas dataframe for use w seaborn
#lupus_pan = pd.DataFrame(lupus_data.X, 
#                         index = lupus_data.obs[["Status", "pop_cov"]], 
#                         columns=lupus_data.var["gene_ids"])


# get observational variables combined
lupus_pan = lupus_data.to_df()

lupus_observations = lupus_data.obs

combo_lupus = lupus_observations.merge(lupus_pan, 
                                       how= "left",
                                       left_index=True,
                                       right_index=True,
                                       validate="one_to_one")


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), # fig size
                     (1, 1) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    print(lupus_data)

    # get observation/variable names
    #lupus_data.obs_names = [f"Cell_{i:d}" for i in range(lupus_data.n_obs)]
    #lupus_data.var_names = [f"Gene_{i:d}" for i in range(lupus_data.n_vars)]
    #print(lupus_data.obs_names[:10])

    # give me all the columns
    pd.set_option('display.max_columns', None)


    print("\n\nOTHER DIMENSIONS: \n\n", lupus_data.obs)
    print("\n\nOTHER DIMENSIONS TYPE: \n\n", type(lupus_data.obs))
    print("\n\n EXTRA COLS: \n\n", list(lupus_data.obs))
    print("\n\n DESCRIBE OUTPUT OF OTHER COLS: \n\n",  lupus_data.obs.describe()) # `ind_cov` has 261 unique values; this is our patient ID
    # running the next line kinda shows why we have to do 
    print("\n\n CELL COUNTS BY PATIENT\n\n", lupus_data.obs.groupby(['ind_cov'])['ind_cov'].count().sort_values())

    print("\n\nGENE IDs: \n\n", lupus_data.var)
    print("\n\nGENE ID TYPE: \n\n", type(lupus_data.var))

    # no missingness <3
    print("missing:", combo_lupus.isnull().sum().describe())


    #print(lupus_pan.head(8))
    print(lupus_observations)
    #print("\n\nCOMBO COMBO: \n\n", combo_lupus)
    #print(type(lupus_pan["Status_pop"][0]))


    


    print("hi aretha")





    return f


