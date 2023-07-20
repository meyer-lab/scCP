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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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

    lupus_tensor, _, group_labs = load_lupus_data(every_n = 10) # don't need to grab cell types here
    


    ranks_to_test = [5,10]

    results = []
    for rank in ranks_to_test:

        # perform pf2 on the given rank
        print('########################################################################\n########################################################################',
              '\n\nPARAFAC2 FITTING: RANK ', str(rank))
        _, factors, _, _ = parafac2_nd(lupus_tensor, 
                                rank = rank, 
                                n_iter_max= 20,
                                random_state = 1, 
                                verbose=True)
        
        A_matrix = factors[0]
        
        # train a logisitic regression model on that rank, using cross validation

        log_reg = LogisticRegression(random_state=0, max_iter = 5000, penalty = 'elasticnet', solver = 'saga')
        parameters = {'l1_ratio':(0, 0.25, 0.5, 0.75, 1), 'C':[1, 10]} # tune penalty and mixture
        grid_search = GridSearchCV(log_reg, parameters)
        grid_search.fit(A_matrix, group_labs.to_numpy())

        # grab fit results as a pandas dataframe, indicate which rank these are from
        initial_results = pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score']]
        initial_results['rank'] = rank

        # expand dictionary (params) into two columns
        results_expanded = initial_results.drop(columns = 'params').join(pd.DataFrame(initial_results['params'].values.tolist(), index=initial_results.index))

        # save best results into results list
        results.append(results_expanded)

    # concatenate all the results into one frame for viewing:

    all_results = pd.concat(results, ignore_index = True)

    print(all_results)
    print(all_results.describe())

    sns.lineplot(data = all_results, 
                 x = 'rank', y = 'mean_test_score', 
                 hue = 'l1_ratio', style = 'C', 
                 palette= 'Set2',
                 ax = ax[0])
    ax[0].set_title('Accuracy by Hyperparameter input')
    ax[0].legend(loc='upper left')

    return f