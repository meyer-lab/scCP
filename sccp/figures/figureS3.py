"""
S3: Logisitic Regression (and maybe SVM) on Pf2 Factor matrix A output
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: Test various Pf2 ranks to see which best predicts disease status

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup
)
from parafac2 import parafac2_nd
from ..imports.scRNA import load_lupus_data
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
import matplotlib

# want to be able to see the different linetypes for this figure
matplotlib.rcParams["legend.handlelength"] = 2

def testPf2Ranks(pfx2_data, condition_labels, ranks_to_test,
                 penalty_type = 'l1', solver = 'saga', error_metric = 'accuracy',
                 penalties_to_test = 10):
    
    results = []
    for rank in ranks_to_test:

        # perform pf2 on the given rank
        print('########################################################################\n########################################################################',
              '\n\nPARAFAC2 FITTING: RANK ', str(rank))
        _, factors, _, _ = parafac2_nd(pfx2_data, 
                                rank = rank, 
                                random_state = 1, 
                                verbose=True)
        
        A_matrix = factors[0]
        
        # train a logisitic regression model on that rank, using cross validation

        log_reg = LogisticRegressionCV(random_state=0, 
                                       max_iter = 5000, 
                                       penalty = penalty_type, 
                                       solver = solver,
                                       Cs = penalties_to_test,
                                       scoring = error_metric)
        
        log_fit = log_reg.fit(A_matrix, condition_labels.to_numpy())

        acc_scores = pd.DataFrame(pd.DataFrame(log_fit.scores_.get('SLE')).mean()).rename(columns = {0: error_metric})
        c_vals = pd.DataFrame(log_fit.Cs_).rename(columns = {0: "penalty"})

        acc_w_c = acc_scores.merge(c_vals, left_index = True, right_index = True)

        # grab fit results as a pandas dataframe, indicate which rank these are from
        initial_results = pd.DataFrame(acc_w_c)
        initial_results['rank'] = rank

        # save best results into results list
        results.append(initial_results)

    # concatenate all the results into one frame for viewing:

    return pd.concat(results, ignore_index = True)

def plotPf2RankTest(rank_test_results, ax, error_metric = "accuracy", palette = 'Set2'):
    sns.lineplot(data = rank_test_results, 
                 x = 'rank', y = error_metric, 
                 hue = 'penalty',
                 palette= 'Set2',
                 ax = ax)
    sns.scatterplot(data = rank_test_results,
                    x = 'rank', y = error_metric,
                    hue = 'penalty',
                    palette= palette,
                    legend=False,
                    ax = ax)
    ax.set_title(error_metric + ' by Hyperparameter input')


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), # fig size
                     (1, 1) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    # rank = 30

    lupus_tensor, _, group_labs = load_lupus_data(every_n=10) # don't need to grab cell types here
    

    ranks_to_test = [5, 10]
    penalties_to_test = [10, 50, 200, 1000]

    results = testPf2Ranks(lupus_tensor, group_labs, ranks_to_test, 
                           penalties_to_test=penalties_to_test)
    plotPf2RankTest(results, ax[0])

    return f