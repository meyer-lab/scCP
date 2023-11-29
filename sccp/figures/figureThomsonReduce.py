"""Downsample Thomson and see how much the factors change.
The goal here is to see how much the dataset size matters."""
import scanpy as sc
from .common import subplotLabel, getSetup
from ..factorization import pf2
from .commonFuncs.plotFactors import (
    plotFactors,
)
from ..imports import import_thomson

from tlviz.factor_tools import factor_match_score as fms
from tensorly.cp_tensor import CPTensor


def makeFigure():
    rank = 20
    X = import_thomson()

    ax, f = getSetup((12, 12), (1, 1))
    subplotLabel(ax)

    # figures for all dataset
    dataX = pf2(X, rank, random_state=1)

    factors = [dataX.uns["Pf2_A"], dataX.uns["Pf2_B"], dataX.varm["Pf2_C"]]
    # plotFactors(dataX, ax[0:3], reorder=(0, 2))

    # sampled_data = sc.pp.subsample(X, fraction=0.99, random_state=0, copy=True)

    samples = [
        sc.pp.subsample(X, fraction=(i / 100), random_state=0, copy=True)
        for i in range(10, 100, 5)
    ]

    # figures for sampled datasets
    # sampledX = pf2(sampled_data, rank, random_state=3)  # type: ignore
    multi_sample = [pf2(sample, rank, random_state=3) for sample in samples]

    # sampled_factors = [
    #     sampledX.uns["Pf2_A"],
    #     sampledX.uns["Pf2_B"],
    #     sampledX.varm["Pf2_C"],
    # ]

    multi_factors = [
        [sampledX.uns["Pf2_A"], sampledX.uns["Pf2_B"], sampledX.varm["Pf2_C"]]
        for sampledX in multi_sample
    ]

    # for sampledX, i in zip(multi_sample, range(1, 20)):
    #     plotFactors(sampledX, ax[3*i:6*i], reorder=(0, 2))

    # factor score match
    dataXcp = CPTensor(
        (
            dataX.uns["Pf2_weights"],
            factors,
        )
    )

    fmsScores = []

    for sampledX, sampled_factors in zip(multi_sample, multi_factors):
        sampledXcp = CPTensor(
            (
                sampledX.uns["Pf2_weights"],
                sampled_factors,
            )
        )
        fmsScores.append(
            fms(dataXcp, sampledXcp, consider_weights=True, skip_mode=None)
        )
        # fmsScore = fms(dataXcp, sampledXcp, consider_weights=True, skip_mode=None)

    ax[0].plot(range(10, 100, 5), fmsScores)
    ax[0].set_title("Factor Match Score for 10-95% percent of data")
    ax[0].set_xlabel("Percent of data")
    ax[0].set_ylabel("Factor Match Score")

    return f
