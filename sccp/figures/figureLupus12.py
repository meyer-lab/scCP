"""
Lupus: Plot factor weights correlations for donor SLE prediction
"""
from anndata import read_h5ad
import itertools
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from .common import subplotLabel, getSetup
from .commonFuncs.plotLupus import getSamplesObs
from ..factorization import correct_conditions
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn import metrics
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing
from sklearn.model_selection import RepeatedStratifiedKFold
from .common import subplotLabel, getSetup


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((5, 2), (1, 1))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    status = getSamplesObs(X.obs)

    Lupus_comp_scan_plot(ax[0],  correct_conditions(X), status)
    # X.obsm["projections"]


    return f


def Lupus_comp_scan_plot(ax, X, status_DF):
    """Plot factor weights for donor SLE prediction"""
    lrmodel = LogisticRegression(penalty=None)
    y = preprocessing.label_binarize(
        status_DF.SLE_status, classes=["Healthy", "SLE"]
    ).flatten()
    
    df = pd.DataFrame([])
    for i in range(X.shape[1]):
        lr = lrmodel.fit(X[:,i].reshape(-1, 1), y)
        score = lr.score(X[:,i].reshape(-1, 1), y)
        df = pd.concat([df, pd.DataFrame({"Component": [i+1], "Weight": [score]})])
        
    sns.barplot(data=df, x="Component", y="Weight", ax=ax)
    ax.set(ylim=(0.5, .8))
    print(df)


    
    # all_comps = np.arange(X.shape[1])
    # acc = np.zeros((X.shape[1], X.shape[1]))

    # for comps in itertools.product(all_comps, all_comps):
    #     if comps[0] >= comps[1]:
    #         compFacs = X[:, [comps[0], comps[1]]]
    #         LR_CoH = lrmodel.fit(compFacs, y)
    #         # print(np.cov(compFacs, rowvar=False))
      
    #         acc[comps[0], comps[1]] = LR_CoH.score(compFacs, y)
    #         acc[comps[1], comps[0]] = acc[comps[0], comps[1]]

    # mask = np.triu(np.ones_like(acc, dtype=bool))

    # cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # ax.set(xlabel="First Component", ylabel="Second Component")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    
    # def ROC_plot(recDF, receptors, cells, tFacDF, comp, ax):
    # """Plots accuracy of classification using receptors and a tfac component"""
    # cv = RepeatedStratifiedKFold(n_splits=10)
    # lrmodel = LogisticRegressionCV(penalty="l1", solver="saga", max_iter=5000, tol=1e-6, cv=cv)
    # AUC_DF = pd.DataFrame()

       
    # predDF = tFacDF[str(comp)].reset_index()
    # predDF.columns = ["Patient", "Comp. " + str(comp)]
    # predDF = predDF.set_index("Patient").join(status_DF.set_index("Patient"), on="Patient")
    # Donor_CoH_y = preprocessing.label_binarize(predDF.Status.values, classes=['Healthy', 'BC']).flatten()
    # LR_CoH = lrmodel.fit(stats.zscore(predDF["Comp. " + str(comp)][:, np.newaxis]), Donor_CoH_y)
    # y_pred = LR_CoH.predict_proba(stats.zscore(predDF["Comp. " + str(comp)][:, np.newaxis]))[:, 1]
    # fpr, tpr, _ = metrics.roc_curve(Donor_CoH_y, y_pred)
    # auc = round(metrics.roc_auc_score(Donor_CoH_y, y_pred), 4)
    # ax.plot(fpr, tpr, label="Comp. " + str(comp) + ", AUC=" + str(auc))
    # AUC_DF = pd.concat([AUC_DF, pd.DataFrame({"Feature": "Comp. " + str(comp), "AUC": [auc]})])
    
#     ax.legend()
#     return AUC_DF


# def plot_AUC_bar(AUC_DF, ax):
#     """Plots AUC from AUC analysis"""
#     sns.barplot(data=AUC_DF, x="Feature", y="AUC", ax=ax)
#     ax.set(ylim=(0.5, 1))
