import pandas as pd
import numpy as np
from parafac2 import parafac2_nd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score



def testPf2Ranks(pfx2_data, condition_labels, ranks_to_test,
                 penalty_type = 'l1', solver = 'saga', error_metric = 'accuracy',
                 penalties_to_test = 10):
    """Tests various numbers of components for Pf2 by optimizing some error metric in logisitic regression (predicting SLE status)"""
    
    results = []
    for rank in ranks_to_test:

        # perform pf2 on the given rank
        print('########################################################################\n',
              '########################################################################',
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

def getCompContribs(A_matrix, target, penalty_amt):
    """Fit logistic regression model, return coefficients of that model"""
    rank = A_matrix.shape[1]
    log_reg = LogisticRegression(random_state=0, max_iter = 5000, penalty = 'l1', solver = 'saga', C = penalty_amt)

    log_fit = log_reg.fit(A_matrix, target)

    coefs = pd.DataFrame(log_fit.densify().coef_,
                         columns = [f"comp_{i}" for i in np.arange(1, rank + 1)]).melt(var_name = "Component",
                                                                                       value_name = "Weight")
    return coefs

def getPf2ROC(A_matrix, conditions, condition_batch_labels, rank, penalties_to_test = 10):
    """Train a logistic regression model using CV on some cohorts, test on another"""
    A_matrix = pd.DataFrame(A_matrix, 
                            index = conditions,
                            columns = [f"comp_{i}" for i in np.arange(1, rank + 1)])
    comps_w_sle_status = A_matrix.merge(condition_batch_labels, left_index=True, right_index=True)
    cohort_3 = comps_w_sle_status[comps_w_sle_status["Processing_Cohort"] == str(3.0)]
    cohorts_124 = comps_w_sle_status[comps_w_sle_status["Processing_Cohort"] != str(3.0)]
    last_comp = "comp_" + str(rank)
    cmp_train = cohorts_124.loc[:, "comp_1":last_comp].to_numpy()
    y_train = cohorts_124.loc[:, "SLE_status"].to_numpy()
    cmp_test = cohort_3.loc[:, "comp_1":last_comp].to_numpy()
    y_test = cohort_3.loc[:, "SLE_status"].to_numpy()
    # train a logisitic regression model using cross validation
    log_reg = LogisticRegressionCV(random_state=0, max_iter = 10000, penalty = 'l1', solver = 'saga',
                                   scoring = "roc_auc",
                                    Cs = penalties_to_test)
    log_fit = log_reg.fit(cmp_train, y_train)

    # get decision function for ROC AUC
    sle_decisions = log_fit.decision_function(cmp_test)
    # validate the ROC AUC of the model
    roc_auc = roc_auc_score(y_test, sle_decisions)
    print("The best ROC AUC is: ", roc_auc)

    return y_test, sle_decisions