"""
Test prediction metrics for Lupus
"""
import numpy as np
from ..logisticReg import testPf2Ranks, getPf2ROC
from sklearn.metrics import roc_auc_score
from ..figures.common import openPf2


def test_accuracy_lupus():
    """Test for accuracy for lupus"""
    rank=40
    accuracyThreshold = .3
    X = openPf2(rank, "Lupus")
    condStatus = X.obs[
        ["Condition", "SLE_status", "Processing_Cohort"]
    ].drop_duplicates()
    condStatus = condStatus.set_index("Condition")

    accuracyResults = testPf2Ranks(
        X,
        condStatus,
        [40],
        cv_group="Processing_Cohort",
    )
    assert np.max(accuracyResults["Accuracy"]) > accuracyThreshold
    
def test_Lupus():
    """Test ROC AUC for Lupus"""
    rank=40
    rocaucThreshold = .9
    X = openPf2(rank, "Lupus")
      
    condStatus = X.obs[
        ["Condition", "SLE_status", "Processing_Cohort", "patient"]
    ].drop_duplicates()
    condStatus = condStatus.set_index("Condition")

    y_test, sle_decisions = getPf2ROC(X.uns["Pf2_A"], condStatus, rank)
    
    assert roc_auc_score(y_test, sle_decisions) > rocaucThreshold
    
    
  


