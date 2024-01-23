import scanpy as sc
from .common import subplotLabel, getSetup
from ..factorization import pf2
from .commonFuncs.plotFactors import (
    plotConditionsFactors,
    plotCellState,
    plotGeneFactors,
    plotWeight,
)
from ..imports import import_thomson

from tlviz.factor_tools import factor_match_score as fms
from tensorly.cp_tensor import CPTensor
from sklearn.metrics import r2_score
import numpy as np

def makeFigure():
    rank = 20
    X = import_thomson()

    # ax, f = getSetup((20,50), (12, 4))
    ax, f = getSetup((20,15), (3, 4))

    subplotLabel(ax)

    dataX = pf2(X, rank, random_state=1)

    factors = [dataX.uns["Pf2_A"], dataX.uns["Pf2_B"], dataX.varm["Pf2_C"]]

    sampled_data = sc.pp.subsample(X, fraction=0.99, random_state=0, copy=True)

    # factor score match
    dataXcp = CPTensor(
        (
            dataX.uns["Pf2_weights"],
            factors,
        )
    )

    percent = list(range(0,11,1))
    fmsScores1 = [1]
    fmsScores2 = [1]
    fmsScores3 = [1]

    for j in range(1,4,1):
        for i in range(1,11,1):
            sampled_data = sc.pp.subsample(X, fraction=1-(i/100), random_state=j, copy=True)
            sampledX = pf2(sampled_data, rank, random_state=3)  # type: ignore
            sampled_factors = [
                sampledX.uns["Pf2_A"],
                sampledX.uns["Pf2_B"],
                sampledX.varm["Pf2_C"],
            ]

            sampledXcp = CPTensor(
            (
                sampledX.uns["Pf2_weights"],
                sampled_factors,
            )
            )

            if j == 1:
                fmsScore = fms(dataXcp, sampledXcp, consider_weights=True, skip_mode=None)
                fmsScores1.append(fmsScore)
            if j == 2:
                fmsScore = fms(dataXcp, sampledXcp, consider_weights=True, skip_mode=None)
                fmsScores2.append(fmsScore)
            if j == 3:
                fmsScore = fms(dataXcp, sampledXcp, consider_weights=True, skip_mode=None)
                fmsScores3.append(fmsScore)

    #percent dropped vs fms graph
    ax[0].plot(percent, fmsScores1, color='pink', label='run 1')
    ax[0].plot(percent, fmsScores2, color='green', label='run 2')
    ax[0].plot(percent, fmsScores3, color='red', label='run 3')
    ax[0].set_xlabel("Percentage of data dropped")
    ax[0].set_ylabel("FMS")
    ax[0].legend()

    return f