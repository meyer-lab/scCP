"""
S3b: Logistic Regression (and maybe SVM) on Pf2 Factor matrix A output
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: run logisitc regression to see which components are best able to predict disease status

# load functions/modules ----
import itertools
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from .common import subplotLabel, getSetup, openPf2
from ..imports.scRNA import load_lupus_data


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 6), (1, 1))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    _, factors, _ = openPf2(rank=40, dataName="lupus", optProjs=True)
    _, obs = load_lupus_data()

    status = obs[["sample_ID", "SLE_status", "Processing_Cohort"]].drop_duplicates()

    Lupus_comp_scan_plot(ax[0], factors[0], status)

    return f


def Lupus_comp_scan_plot(ax, X, status_DF):
    """Plot factor weights for donor BC prediction"""
    lrmodel = LogisticRegression(penalty=None)
    y = preprocessing.label_binarize(
        status_DF.SLE_status, classes=["Healthy", "SLE"]
    ).flatten()
    all_comps = np.arange(X.shape[1])
    acc = np.zeros((X.shape[1], X.shape[1]))

    for comps in itertools.product(all_comps, all_comps):
        if comps[0] >= comps[1]:
            compFacs = X[:, [comps[0], comps[1]]]
            LR_CoH = lrmodel.fit(compFacs, y)
            acc[comps[0], comps[1]] = LR_CoH.score(compFacs, y)
            acc[comps[1], comps[0]] = acc[comps[0], comps[1]]

    sns.heatmap(
        data=acc,
        vmin=0.5,
        vmax=1,
        xticklabels=all_comps + 1,
        yticklabels=all_comps + 1,
        cbar_kws={"label": "Classification Accuracy"},
        ax=ax,
    )

    ax.set(xlabel="First Component", ylabel="Second Component")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)