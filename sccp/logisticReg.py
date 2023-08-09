import pandas as pd
import numpy as np
from parafac2 import parafac2_nd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

def testPf2Ranks(pfx2_data, condition_labels_all, ranks_to_test,
                 penalty_type = 'l1', solver = 'saga', error_metric = 'accuracy',
                 penalties_to_test = 10, cv_group = None):
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

        condition_labels = condition_labels_all['SLE_status']
        
        # train a logisitic regression model on that rank, using cross validation
        # if we want certain cross validation groups; make them

        
        if cv_group == None:
            log_reg = LogisticRegressionCV(random_state=0, 
                                       max_iter = 5000, 
                                       penalty = penalty_type, 
                                       solver = solver,
                                       Cs = penalties_to_test,
                                       scoring = error_metric)
        else:
            sgkf = StratifiedGroupKFold(n_splits=4)
            group_cond_labels = condition_labels_all[cv_group]

            log_reg = LogisticRegressionCV(random_state=0, 
                                        max_iter = 5000, 
                                        cv= sgkf.split(A_matrix, condition_labels.to_numpy(), group_cond_labels.to_numpy()),
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