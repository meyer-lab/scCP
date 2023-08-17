"""
S4b: Using Original Data to look at (possible) Megakaryocytes
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: look at distribution of expression of platelet/MK cell genes 

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup,
    flattenData
)
from ..imports.scRNA import load_lupus_data

import seaborn as sns
import pandas as pd
import anndata
import numpy as np

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 6), # fig size
                     (2, 2) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/lupus/lupus.h5ad")

    # rename columns to make more sense 
    X.obs = X.obs.rename({'batch_cov': 'pool',
                          'ind_cov': 'patient',
                          'cg_cov': 'cell_type_broad',
                          'ct_cov': 'cell_type_lympho',
                          'ind_cov_batch_cov': 'sample_ID',
                          'Age': 'age',
                          'Sex': 'sex',
                          'pop_cov': 'ancestry',
                          'Status': 'SLE_condition'}, axis = 1)
    
    # get rid of IGTB1906_IGTB1906:dmx_count_AHCM2CDMXX_YE_0831 (only 3 cells)
    X = X[X.obs['sample_ID'] != 'IGTB1906_IGTB1906:dmx_count_AHCM2CDMXX_YE_0831']

    # reorder X so that all of the patients are in alphanumeric order. this is important
    # so that we can steal cell typings at this point
    obsV = X.obs_vector('sample_ID')
    sgUnique, sgIndex = np.unique(obsV, return_inverse=True)

    ann_data_objects = [X[sgIndex == sgi, :] for sgi in range(len(sgUnique))]

    X = anndata.concat(ann_data_objects, axis=0)

    data = pd.DataFrame(X.X, index = X.obs_names, columns=X.var_names)
    groups = pd.DataFrame(X.obs[['SLE_status', 'louvain']])

    merged = data.merge(groups, left_index=True, right_index=True).reset_index(drop = True)

    # get only louvain 14: these are probably our megakaryocytes

   #merged = merged[merged['louvain'] == str(14)]

    print(merged)

    genes = ['PPBP', 'PF4', 'TUBB1', 'CLU']

    for i in range(len(genes)):
        sns.boxplot(data = merged, x = genes[i], y = 'louvain', showfliers = False, ax = ax[i])
        ax[i].set_title('Expression of ' + genes[i] + ' by louvain cluster')


    return f