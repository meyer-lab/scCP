"""Downsample Thomson and see how much the factors change.
The goal here is to see how much the dataset size matters."""
import scanpy as sc
from .common import subplotLabel, getSetup
from ..factorization import pf2
from .commonFuncs.plotFactors import (
    plotConditionsFactors,
    plotCellState,
    plotGeneFactors,
)
from ..imports import import_thomson
from tlviz.factor_tools import factor_match_score as fms
from tensorly.cp_tensor import CPTensor
from .figureThomson1 import groupDrugs


def makeFigure():
    rank = 20
    X = import_thomson()

    ax, f = getSetup((12, 12), (2, 3))
    subplotLabel(ax)

    # figures for all dataset
    dataX = pf2(X, rank, random_state=1, doEmbedding=False)

    plotConditionsFactors(
        dataX, ax[0], groupDrugs(dataX.obs["Condition"]), ThomsonNorm=True
    )
    plotCellState(dataX, ax[1])
    plotGeneFactors(dataX, ax[2])
    dataXcp = CPTensor(
        (
            dataX.uns["Pf2_weights"],
            [dataX.uns["Pf2_A"], dataX.uns["Pf2_B"], dataX.varm["Pf2_C"]],
        )
    )

    sampled_X = sc.pp.subsample(X, fraction=0.99, random_state=0, copy=True)
    sampledX = pf2(sampled_X, rank, random_state=3, doEmbedding=False)

    samp_cp = CPTensor(
        (
            sampledX.uns["Pf2_weights"],
            [sampledX.uns["Pf2_A"], sampledX.uns["Pf2_B"], sampledX.varm["Pf2_C"]],
        )
    )

    # factor score match
    fmsScore, perm = fms(
        dataXcp,
        samp_cp,
        consider_weights=True,
        skip_mode=1,
        return_permutation=True,
    )
    print(fmsScore)

    # Permute the components so we can see how they are matched up
    sampledX.uns["Pf2_weights"] = sampledX.uns["Pf2_weights"][perm]
    sampledX.uns["Pf2_A"] = sampledX.uns["Pf2_A"][:, perm]
    sampledX.uns["Pf2_B"] = sampledX.uns["Pf2_B"][:, perm]
    sampledX.varm["Pf2_C"] = sampledX.varm["Pf2_C"][:, perm]  # type: ignore

    plotConditionsFactors(
        sampledX, ax[3], groupDrugs(dataX.obs["Condition"]), ThomsonNorm=True
    )
    plotCellState(sampledX, ax[4])
    plotGeneFactors(sampledX, ax[5])

    return f
