"""
Lupus: Plot logistic regression weights for SLE and/or ancestry
"""
from anndata import read_h5ad
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_decomposition import PLSRegression 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import CategoricalNB
from .common import subplotLabel, getSetup
from .commonFuncs.plotLupus import getSamplesObs
from ..factorization import correct_conditions
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



def getCompContribs(X: np.ndarray, y: pd.Series) -> pd.DataFrame:
    """Fit logistic regression model, return coefficients of that model"""
    lr = LogisticRegressionCV(
        random_state=0, max_iter=100000, penalty="l1", solver="saga"
    ).fit(X, y)

    cmp_col = [i for i in range(1, X.shape[1] + 1)]
    coefs = pd.DataFrame({"Component": cmp_col, "Weight": lr.coef_.flatten()})
    print(f"LogRig Score: {lr.score(X, y)}")

    return coefs


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((5, 8), (4, 1))

    # Add subplot labels
    subplotLabel(ax)

    data = read_h5ad("/opt/andrew/lupus/lupus_fitted.h5ad")

    data.uns["Pf2_A"] = correct_conditions(data)

    y = getSamplesObs(data.obs)
    y = y["SLE_status"]
    y = y.to_numpy()
    X = np.array(data.uns["Pf2_A"])

    contribsStatus = getCompContribs(X, y)
    sns.barplot(
        data=contribsStatus,
        x="Component",
        y="Weight",
        color="k",
        errorbar=None,
        ax=ax[0],
    )
    # ax[0].tick_params(axis="x", rotation=90)
    ax[0].set_title("LogReg")

    
    naivebayes = CategoricalNB()
    naivebayes.fit(X, y)
    print("Categorical Native Bayes Score:", naivebayes.score(X, y))
      
    lda = LDA(n_components=1)
    lda.fit(X, y)
    print("LDA Score:", lda.score(X, y))
    
    cmp_col = [i for i in range(1, X.shape[1] + 1)]
    coefs = pd.DataFrame({"Component": cmp_col, "Weight": lda.coef_.flatten()})
    newX = lda.fit_transform(X, y)

    
    sns.barplot(
        data=coefs,
        x="Component",
        y="Weight",
        color="k",
        errorbar=None,
        ax=ax[1],
    )
    ax[1].set_title("LDA")
    
    sns.histplot(x=newX.flatten(), element="step", hue=y, ax=ax[2])
    ax[2].set(xlabel="LDA 1")
    ax[2].set_title("LDA")
    
    
    sm = svm.SVC(kernel="linear")
    fittedsvm = sm.fit(X,y)
    print("SVM Score:", sm.score(X,y))
    
    coefs = pd.DataFrame({"Component": cmp_col, "Weight": fittedsvm.coef_.flatten()})
    sns.barplot(
        data=coefs,
        x="Component",
        y="Weight",
        color="k",
        errorbar=None,
        ax=ax[3],
    )
    ax[3].set_title("SVM")
 
    
    plsr = PLSRegression(n_components=2, scale=False)
    y[y == "Healthy"] = 0
    y[y == "SLE"] = 1
    plsr.fit(X, y)
    print("PLS-DA Score:", plsr.score(X, y))
    
  
    

    return f
