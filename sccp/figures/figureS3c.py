"""
S3: Logisitic Regression (and maybe SVM) on Pf2 Factor matrix A output
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: run logisitc regression and assess predictive ability of 39 comp Pf2 using ROC AUC

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup,
    openPf2
)
from ..imports.scRNA import load_lupus_data
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, RocCurveDisplay


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 6), # fig size
                     (1, 1) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    lupus_tensor, _, group_labs = load_lupus_data(give_batch=True) 

    patients = lupus_tensor.condition_labels
    
    _, factors, _, = openPf2(rank = 39, dataName = 'lupus')

        
    A_matrix = pd.DataFrame(factors[0], 
                            index = patients,
                            columns = [f"comp_{i}" for i in np.arange(1, 40)])
    
    comps_w_sle_status = A_matrix.merge(group_labs, left_index=True, right_index=True)

    cohort_4 = comps_w_sle_status[comps_w_sle_status["Processing_Cohort"] == str(4.0)]
    cohorts_123 = comps_w_sle_status[comps_w_sle_status["Processing_Cohort"] != str(4.0)]

    cmp_train = cohort_4.loc[:, "comp_1":"comp_39"].to_numpy()
    y_train = cohort_4.loc[:, "SLE_status"].to_numpy()
    cmp_test = cohorts_123.loc[:, "comp_1":"comp_39"].to_numpy()
    y_test = cohorts_123.loc[:, "SLE_status"].to_numpy()

    
    # SPLIT DATA INTO TEST/TRAIN: want to get 

    #cmp_train, cmp_test, y_train, y_test = train_test_split(A_matrix, group_labs.to_numpy(), 
    #                                                        stratify=group_labs.to_numpy(),
    #                                                        test_size = 0.75,
    #                                                        random_state=0)
#
 
    # train a logisitic regression model using cross validation

    log_reg = LogisticRegressionCV(random_state=0, max_iter = 5000, penalty = 'l1', solver = 'saga',
                                   scoring = "roc_auc",
                                    Cs = [2, 10, 20, 30, 50, 100, 150, 200, 1000])
    log_fit = log_reg.fit(cmp_train, y_train)

    # get decision function for ROC AUC
    sle_decisions = log_fit.decision_function(cmp_test)

    # validate the ROC AUC of the model

    roc_auc = roc_auc_score(y_test, sle_decisions)

    print("The best ROC AUC is: ", roc_auc)

    RocCurveDisplay.from_predictions(y_test, sle_decisions, 
                                     pos_label = "SLE",
                                     plot_chance_level = True,
                                     ax = ax[0])
    
    ax[0].set_title('OOS ROC for Cases/Controls: 39 Comp LASSO')

    return f