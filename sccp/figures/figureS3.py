"""
S3: Logisitic Regression (and maybe SVM) on Pf2 Factor matrix A output
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: run logisitc regression to see which components are best able to predict disease status

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup
)
from ..parafac2 import parafac2_nd
from ..imports.scRNA import load_lupus_data
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
import matplotlib

# want to be able to see the different linetypes for this figure
matplotlib.rcParams["legend.handlelength"] = 2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), # fig size
                     (1, 1) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    # rank = 30

    lupus_tensor, _, group_labs = load_lupus_data() # don't need to grab cell types here
    

    ranks_to_test = list(range(39,40))

    results = []
    for rank in ranks_to_test:

        # perform pf2 on the given rank
        print('########################################################################\n########################################################################',
              '\n\nPARAFAC2 FITTING: RANK ', str(rank))
        _, factors, _, _ = parafac2_nd(lupus_tensor, 
                                rank = rank, 
                                random_state = 1, 
                                verbose=True)
        
        A_matrix = factors[0]
        
        # train a logisitic regression model on that rank, using cross validation

        log_reg = LogisticRegressionCV(random_state=0, max_iter = 5000, penalty = 'l1', solver = 'saga',
                                       Cs = [2, 10, 20, 30, 50, 100, 150, 200, 1000])
        log_fit = log_reg.fit(A_matrix, group_labs.to_numpy())

        acc_scores = pd.DataFrame(pd.DataFrame(log_fit.scores_.get('SLE')).mean()).rename(columns = {0: "accuracy"})
        print(acc_scores)
        c_vals = pd.DataFrame(log_fit.Cs_).rename(columns = {0: "penalty"})
        print(c_vals)

        acc_w_c = acc_scores.merge(c_vals, left_index = True, right_index = True)
        print(acc_w_c)


        #print(log_fit.Cs_)

        # grab fit results as a pandas dataframe, indicate which rank these are from
        initial_results = pd.DataFrame(acc_w_c)
        initial_results['rank'] = rank

        # expand dictionary (params) into two columns
        #results_expanded = initial_results.drop(columns = 'params').join(pd.DataFrame(initial_results['params'].values.tolist(), index=initial_results.index))

        # save best results into results list
        results.append(initial_results)

    # concatenate all the results into one frame for viewing:

    all_results = pd.concat(results, ignore_index = True)

    all_results.to_csv('/home/seanp/scCP-2/sccp/data/logreg_results.csv')

    print(all_results)
    print(all_results.describe())

    sns.lineplot(data = all_results, 
                 x = 'rank', y = 'accuracy', 
                 hue = 'penalty',
                 palette= 'Set2',
                 ax = ax[0])
    ax[0].set_title('Accuracy by Hyperparameter input')
    #ax[0].legend(loc='upper left')

    return f