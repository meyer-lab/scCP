"""
Test prediction metrics for Lupus
"""
import anndata
from ..logisticReg import getPf2ROC
from sklearn.metrics import roc_auc_score
from ..figures.commonFuncs.plotLupus import getSamplesObs


def test_Lupus_AUCROC():
    """Test ROC AUC for Lupus"""
    X = anndata.read_h5ad(f"/opt/pf2/Lupus_analyzed_40comps.h5ad", backed="r")

    condStatus = getSamplesObs(X.obs)

    y_test, sle_decisions = getPf2ROC(X.uns["Pf2_A"], condStatus)

    assert roc_auc_score(y_test, sle_decisions) > 0.925
