"""
Lupus: Plot logistic regression weights for SLE and/or ancestry
"""
from anndata import read_h5ad
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from .common import subplotLabel, getSetup
from .commonFuncs.plotLupus import getSamplesObs


def getCompContribs(X: np.ndarray, y: pd.Series) -> pd.DataFrame:
    """Fit logistic regression model, return coefficients of that model"""
    lr = LogisticRegressionCV(
        random_state=0, max_iter=100000, penalty="l1", solver="saga"
    ).fit(X, y)
    
    prediction = lr.predict(X)
    df = pd.DataFrame({"Prediction": prediction, "Weight": y.to_numpy()})
    
    comparison_vector = []
    for i in range(df.shape[0]):
        comparison = df.iloc[i, :].to_numpy()
        if comparison[0] == comparison[1]:
            comparison_vector = np.append(comparison_vector, [1])
        if comparison[0] != comparison[1]:
            comparison_vector = np.append(comparison_vector, [0])
            
    print(df)
    print(comparison_vector)

    # cmp_col = [f"Cmp. {i}" for i in range(1, X.shape[1] + 1)]
    # coefs = pd.DataFrame({"Component": cmp_col, "Weight": lr.coef_.flatten()})
    # print(coefs)
    # print(y)
    # print(f"Fitting accuracy: {lr.score(X, y)}")

    return comparison_vector


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((5, 8), (2, 1))

    # Add subplot labels
    subplotLabel(ax)

    data = read_h5ad("/opt/andrew/lupus/lupus_fitted.h5ad", backed="r")
    # _, count = np.unique(data.obs["Condition"], return_counts=True)
    # # Amatrix= data.uns["Pf2_A"]
    # # for i in range(Amatrix.shape[1]):
    # #     Amatrix[:, i] /= count
        
    # # data.uns["Pf2_A"]= Amatrix

    # df_y = getSamplesObs(data.obs)
    # X = np.array(data.uns["Pf2_A"])

    # contribsStatus = getCompContribs(X, df_y["SLE_status"])
    
    # status = df_y["SLE_status"].values
    
    # print(contribsStatus)
    # sns.barplot(
    #     data=contribsStatus,
    #     x="Component",
    #     y="Weight",
    #     color="k",
    #     errorbar=None,
    #     ax=ax[0],
    # )
    # ax[0].tick_params(axis="x", rotation=90)
    # ax[0].set_title("Weight of Pf2 Cmps in Logsitic Regression: Predicting: SLE")


    # df_y["ancestry"] = df_y["ancestry"] == "European"
    # contribsAnc = getCompContribs(X, df_y["ancestry"])

    # contribsStatus["Predicting"] = "SLE Status"
    # contribsAnc["Predicting"] = "Euro-Ancestry"
    # contribs = pd.concat([contribsStatus, contribsAnc])
    
    
    df_y = getSamplesObs(data.obs)
    X = np.array(data.uns["Pf2_A"])

    contribsStatus = getCompContribs(X, df_y["SLE_status"])
    
    status = df_y["SLE_status"].values
    _, count = np.unique(data.obs["Condition"], return_counts=True)
    
    
    ax[0].scatter(x=contribsStatus, y=count)
    ax[0].set(xlabel=f"0 LR Predicted Incorrectly; LR Predicted Correctly 1", ylabel="Cell Number per Experiment")

    # sns.barplot(
    #     data=contribs,
    #     x="Component",
    #     y="Weight",
    #     hue="Predicting",
    #     errorbar=None,
    #     ax=ax[1],
    # )
    # ax[1].tick_params(axis="x", rotation=90)
    # ax[1].set_title("Weight of Pf2 Cmps in Logsitic Regression")

    return f
