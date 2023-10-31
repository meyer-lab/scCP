"""
Thomson: XX
"""
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
)
from .commonFuncs.plotUMAP import plotLabelsUMAP
from ..gating import gateThomsonCells
import seaborn as sns


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 10), (4, 4))

    # Add subplot labels
    subplotLabel(ax)
    rank = 30
    X = openPf2(rank, "Thomson")
    print(X.uns["Pf2_A"].shape)
    print(X.uns["Pf2_B"].shape)
    print(X.varm["Pf2_C"].shape)

    gateThomsonCells(X)
    # plotLabelsUMAP(X, "Cell Type", ax[0])
    # plotLabelsUMAP(X, "Cell Type2", ax[1])
    
    # X = X[:, ["PF4", "SDPR", "GNG11", "PPBP"]]
    
    print(X.obs["Processing_Cohort"])
    # for

    #     sns.histplot(X[:, "PF4"].X, ax=ax[0], bins=100)
    #     sns.histplot(X[:, "GNG11"].X, ax=ax[1], bins=100)
    #     sns.histplot(X[:, "PPBP"].X, ax=ax[2], bins=100)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')
    #     ax[0].set(xlabel="Gene Expression", title="PF4")
    #     ax[1].set(xlabel="Gene Expression", title="GNG11")
    #     ax[2].set(xlabel="Gene Expression", title="PPBP")
    # sns.histplot(X[:, "SDPR"].X, ax=ax[3])

    # weightedProjDF = flattenWeightedProjs(data, factors[1], projs)
    # weightedProjDF["Cell Type"] = dataDF["Cell Type"].values
    # weightedProjDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    # dataDF.sort_values(by=["Condition", "Cell Type"], inplace=True)

    # comps = [5, 12, 20, 30]
    # for i, comp in enumerate(comps):
    #     plotCmpPerCellType(weightedProjDF, comps[i], ax[(2 * i) + 1], outliers=False)
    #     plotCmpUMAP(comps[i], factors[1], pf2Points, projs, ax[(2 * i) + 2])

    # geneSet1 = ["NKG7", "GNLY", "GZMB", "GZMH", "PRF1"]
    # geneSet2 = ["MS4A1", "CD79A", "CD79B", "TNFRSF13B", "BANK1"]

    # genes = [geneSet1, geneSet2]
    # for i in range(len(genes)):
    #     plotGenePerCellType(genes[i], dataDF, ax[i + 9])

    # set3 = ["VPREB3", "CD79A", "FAM111B", "HOPX", "SLC30A3", "MS4A1"]
    # set4 = ["CD163", "ADORA3", "MS4A6A", "RNASE1", "MTMR11"]

    # glucs = [
    #     "Betamethasone Valerate",
    #     "Loteprednol etabonate",
    #     "Budesonide",
    #     "Triamcinolone Acetonide",
    #     "Meprednisone",
    # ]
    # geneSet3 = ["CD163", "ADORA3"]
    # plotGenePerCategCond(glucs, "Gluco", geneSet3, dataDF, ax[11:13])

    # geneSet4 = ["VPREB3", "FAM111B"]
    # plotGenePerCategCond(
    #     ["Dexrazoxane HCl (ICRF-187, ADR-529)"], "Dex HCl", geneSet4, dataDF, ax[13:15]
    # )

    return f
