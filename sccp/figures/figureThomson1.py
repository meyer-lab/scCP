"""
Thomson: Plotting Pf2 factors and weights
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotFactors import (
    plotFactors,
    plotWeight,
)
import numpy as np
import pandas as pd
from ..imports import import_thomson
from ..parafac2 import pf2
from .commonFuncs.plotUMAP import plotGeneUMAP, plotLabelsUMAP
import pacmap
import numpy as np
from ..gating import gateThomsonCells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 20), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    rank = 30
    # X = openPf2(rank, "Thomson")
    X = import_thomson()
    print(X)
    X = pf2(X, rank)
    print(X)
    drugNames = groupDrugs(X.obs["Condition"])

    factors = [X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]]
    plotFactors(
        factors, X, ax[0:3], reorder=(0, 2), trim=(2,), cond_group_labels=drugNames
    )
    plotWeight(X.uns["Pf2_weights"], ax[3])
    gateThomsonCells(X)
    plotLabelsUMAP(X, "Cell Type", ax[4])
    plotLabelsUMAP(X, "Cell Type2", ax[5])

    return f


def groupDrugs(labels):
    """Groups drugs of similar category"""
    names = np.unique(labels)

    glucs = [
        "Triamcinolone Acetonide",
        "Loteprednol etabonate",
        "Betamethasone Valerate",
        "Budesonide",
        "Meprednisone",
    ]
    for i in glucs:
        names[names == i] = "Glucocoritcoids"

    ctrl = ["CTRL1", "CTRL2", "CTRL3", "CTRL4", "CTRL5", "CTRL6"]
    for i in ctrl:
        names[names == i] = "Control"

    names[names == "Everolimus (RAD001)"] = "Prostaglandin"
    names[names == "Rapamycin (Sirolimus)"] = "Prostaglandin"
    names[names == "Alprostadil"] = "mTOR Inhibitor"
    names[names == "Cyclosporine"] = "Calcineruin Inhibitor"

    condition = [
        "Glucocoritcoids",
        "Control",
        "Prostaglandin",
        "mTOR Inhibitor",
        "Calcineruin Inhibitor",
    ]

    names = pd.Series([c if c in condition else "Other" for c in names])

    return names
