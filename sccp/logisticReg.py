import pandas as pd
import numpy as np
from parafac2 import parafac2_nd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

def getPf2ROC(A_matrix, conditions, condition_batch_labels, rank, penalties_to_test = 10):
    """Train a logistic regression model using CV on some cohorts, test on another"""
    A_matrix = pd.DataFrame(A_matrix, 
                            index = conditions,
                            columns = [f"comp_{i}" for i in np.arange(1, rank + 1)])
    cond_labels = condition_batch_labels.reset_index()[['SLE_status', 'sample_ID']].drop_duplicates().set_index('sample_ID')
    comps_w_sle_status = A_matrix.merge(cond_labels, left_index=True, right_index=True, how = 'left')
    patients_in_batch_4 = condition_batch_labels[condition_batch_labels['Processing_Cohort'] == str(4.0)].index.tolist()
    other_patients = []
    for pat in conditions:
        if pat not in patients_in_batch_4:
            other_patients.append(pat)

    assert len(conditions) == (len(patients_in_batch_4) + len(other_patients))

    cohort_4 = comps_w_sle_status[comps_w_sle_status.index.isin(patients_in_batch_4)]
    cohorts_123 = comps_w_sle_status[comps_w_sle_status.index.isin(other_patients)]
    last_comp = "comp_" + str(rank)
    cmp_train = cohort_4.loc[:, "comp_1":last_comp].to_numpy()
    y_train = cohort_4.loc[:, "SLE_status"].to_numpy()
    cmp_test = cohorts_123.loc[:, "comp_1":last_comp].to_numpy()
    y_test = cohorts_123.loc[:, "SLE_status"].to_numpy()
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