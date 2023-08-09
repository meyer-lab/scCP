import pandas as pd
import numpy as np
from parafac2 import parafac2_nd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

def getCompContribs(A_matrix, target, penalty_amt = 50):
    """Fit logistic regression model, return coefficients of that model"""
    rank = A_matrix.shape[1]
    log_reg = LogisticRegression(random_state=0, max_iter = 5000, penalty = 'l1', solver = 'saga', C = penalty_amt)

    log_fit = log_reg.fit(A_matrix, target)

    coefs = pd.DataFrame(log_fit.densify().coef_,
                         columns = [f"comp_{i}" for i in np.arange(1, rank + 1)]).melt(var_name = "Component",
                                                                                       value_name = "Weight")
    return coefs