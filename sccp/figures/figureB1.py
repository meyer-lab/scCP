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
    X = import_thomson()

    ax, f = getSetup((10,5), (1, 2))

    subplotLabel(ax)

    ranks = [5,10,15,20,25,30,35,40,45,50]
    fmsList = []

# testing different ranks
    for i in ranks:
        dataX = pf2(X, i, random_state=1)
        factors = [dataX.uns["Pf2_A"], dataX.uns["Pf2_B"], dataX.varm["Pf2_C"]]
        dataXcp = CPTensor(
        (
            dataX.uns["Pf2_weights"],
            factors,
        )
        )
        
        sampled_data = sc.pp.subsample(X, fraction=0.99, random_state=1, copy=True)
        sampledX = pf2(sampled_data, i, random_state=2)  # type: ignore
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
        
        fmsScore = fms(dataXcp, sampledXcp, consider_weights=True, skip_mode=1)
        fmsList.append(fmsScore)

    #rank vs fms graph
    ax[0].plot(ranks, fmsList, color='pink')
    
    ax[0].set_xlabel("Rank")
    ax[0].set_ylabel("FMS")

    return f