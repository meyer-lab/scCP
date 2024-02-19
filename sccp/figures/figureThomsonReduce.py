import scanpy as sc
import anndata
from .common import subplotLabel, getSetup
from ..factorization import pf2
from ..imports import import_thomson
from matplotlib.axes import Axes
from tlviz.factor_tools import factor_match_score as fms
from tensorly.cp_tensor import CPTensor
import seaborn as sns


def plotFMSpercentDrop(
    X: anndata.AnnData,
    ax: Axes,
    percentDropped: int,
    runs: int,
    step: int,
    rank=20,
):
    # pf2 on original dataset
    dataX = pf2(X, rank, random_state=1)
    factors = [dataX.uns["Pf2_A"], dataX.uns["Pf2_B"], dataX.varm["Pf2_C"]]

    dataXcp = CPTensor(
        (
            dataX.uns["Pf2_weights"],
            factors,
        )
    )

    percentList = range(0, percentDropped + 1, step)
    fmsLists = []

    # loop to do multiple runs
    for j in range(0, runs, 1):
        scores = [1]

        # loop to compare sampled dataset to original
        for i in percentList[1:]:
            sampled_data = sc.pp.subsample(
                X, fraction=1 - (i / 100), random_state=j + 1, copy=True
            )
            sampledX = pf2(sampled_data, rank, random_state=j + 1)  # type: ignore
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
            scores.append(fmsScore)
        fmsLists.append(scores)

    random_colors = sns.color_palette("husl", len(fmsLists))
    for n in range(0, runs, 1):
        ax.plot(percentList, fmsLists[n], label=f"Run {n+1}", color=random_colors[n])

    # percent dropped vs fms graph
    ax.set_xlabel("Percentage of Data Dropped")
    ax.set_ylabel("FMS")
    ax.set_title("Percent of Data Dropped vs FMS")
    ax.legend()


def plotRankTest(
    X: anndata.AnnData,
    ax: Axes,
    percentDrop: float,
    ranksList: list[int],
):
    fmsList = []

    # testing different ranks input into function with one percent valued dropped
    for i in ranksList:
        dataX = pf2(X, rank=i, random_state=i)
        factors = [dataX.uns["Pf2_A"], dataX.uns["Pf2_B"], dataX.varm["Pf2_C"]]
        dataXcp = CPTensor(
            (
                dataX.uns["Pf2_weights"],
                factors,
            )
        )

        sampled_data = sc.pp.subsample(
            X, fraction=percentDrop, random_state=i, copy=True
        )
        sampledX = pf2(sampled_data, rank=i, random_state=i)  # type: ignore
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

    # rank vs fms graph
    ax.plot(ranksList, fmsList, color="pink")
    ax.set_xlabel("Rank")
    ax.set_ylabel("FMS")
    ax.set_title("Rank vs FMS")


# testing functions
def makeFigure():
    X = import_thomson()

    ax, f = getSetup((15, 5), (1, 3))

    subplotLabel(ax)

    plotFMSpercentDrop(X, ax[0], percentDropped=10, runs=3, step=1)

    ranks = range(5, 51, 5)
    plotRankTest(X, ax[2], percentDrop=0.99, ranksList=ranks)

    return f
