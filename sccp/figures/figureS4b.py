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
    ax, f = getSetup((10, 5), # fig size
                     (1,2) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    lupus_tensor, obs = load_lupus_data()

    groups = obs[['sample_ID','SLE_status']].reset_index(drop = True).drop_duplicates()

    lupus_flat = flattenData(lupus_tensor)

    # step 2: figure out whats up with MHC class II 

    lupus_flat['mhc_index'] = (lupus_flat['HLA-DPA1'] + lupus_flat['HLA-DPB1'] + lupus_flat['HLA-DQA1'] + lupus_flat['HLA-DQB1'] + lupus_flat['HLA-DRA'] + lupus_flat['HLA-DRB1'])/6

    avg_mhc_index = lupus_flat.groupby('Condition')['mhc_index'].mean().reset_index()

    lupus_flat['ifit_index'] = (lupus_flat['IFIT1'] + lupus_flat['IFIT2'] + lupus_flat['IFIT3'])/3

    avg_ifit_index = lupus_flat.groupby('Condition')['ifit_index'].mean().reset_index()


    avg_pf4 = lupus_flat.groupby('Condition')['PF4'].mean().reset_index()
    print(avg_mhc_index)

    #lupus_flat_sle = lupus_flat.merge(groups, right_index = True, left_index = True)

    # step one: figure out which cells have high MK marker gene counts

    #lupus_flat_sle = lupus_flat_sle[lupus_flat_sle['PPBP'] > 0]
    genes = ['PPBP', 'PF4', 'PTCRA']

    #for i, gene in enumerate(genes):
    #    sns.histplot(data = lupus_flat, x = gene, bins = 50, ax = ax[i+1])
   
   # just for PPBP:
    lupus_flat['MK'] = np.where((lupus_flat['PF4'] > -0.1) & (lupus_flat['PPBP'] > -0.1) & (lupus_flat['PTCRA'] > -0.1),
                                 True, 
                                 False)

    samples = lupus_flat.groupby(['Condition', 'MK']).count().reset_index()

    samples = samples.rename({'PPBP': 'MK_count'}, axis = 1)[['Condition', 'MK', 'MK_count']]

    samples_wide = samples.pivot(columns = 'MK', index = 'Condition', values = 'MK_count').reset_index()

    samples_SLE = samples_wide.merge(groups, left_on = 'Condition', right_on = 'sample_ID')
    samples_SLE[True] = np.where(samples_SLE[True].isnull(), 0, samples_SLE[True])
    print(samples_SLE)

    samples_SLE['pct_MKs'] = samples_SLE[True]/(samples_SLE[True] + samples_SLE[False])
    
    samples_SLE['pct_MKs'] *= 100


    samples_SLE_mhc = samples_SLE.merge(avg_mhc_index, left_on = "Condition", right_on = 'Condition')
    samples_SLE_mhc_pf4 = samples_SLE_mhc.merge(avg_pf4, left_on = "Condition", right_on = 'Condition')
    samples_SLE_mhc_pf4_ifit = samples_SLE_mhc_pf4.merge(avg_ifit_index, left_on = "Condition", right_on = 'Condition')
    print(samples_SLE_mhc)

    sns.scatterplot(data = samples_SLE_mhc_pf4_ifit, y = 'mhc_index', x = 'pct_MKs', hue = 'SLE_status', ax = ax[1])
    sns.scatterplot(data = samples_SLE_mhc_pf4_ifit, y = 'mhc_index', x = 'ifit_index', hue = 'SLE_status', ax = ax[0])
    #sns.swarmplot(data = samples_SLE_mhc_pf4_ifit, y = 'mhc_index', hue = 'SLE_status', ax = ax[0])

   # for i in range(len(genes)):
   #     sns.boxplot(data = merged, x = genes[i], y = 'louvain', showfliers = False, ax = ax[i])
    #    ax[i].set_title('Expression of ' + genes[i] + ' by louvain cluster')


    return f