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


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 6), (1, 1))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    status = getSamplesObs(X.obs)

    Lupus_comp_scan_plot(ax[0], correct_conditions(X), status)


    return f


def Lupus_comp_scan_plot(ax, X, status_DF):
    """Plot factor weights for donor SLE prediction"""
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
            # print(np.cov(compFacs, rowvar=False))
      
            acc[comps[0], comps[1]] = pearsonr(compFacs[:, 0],compFacs[:, 1])[0]
            acc[comps[1], comps[0]] = acc[comps[0], comps[1]]

    mask = np.triu(np.ones_like(acc, dtype=bool))

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(
        data=acc,
        vmax=1,
        vmin=-1,
        mask=mask,
        xticklabels=all_comps + 1,
        yticklabels=all_comps + 1,
        cbar_kws={"label": "Pearson Coefficient"},
        cmap=cmap,
        ax=ax,
    )

    ax.set(xlabel="First Component", ylabel="Second Component")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
