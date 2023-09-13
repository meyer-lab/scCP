"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
    openUMAP,
    flattenData,
    flattenWeightedProjs,
)
import matplotlib
from matplotlib import gridspec, pyplot as plt
from .commonFuncs.plotGeneral import plotGenePerCellType, plotGenePerCategCond
from .commonFuncs.plotUMAP import (
    plotCellTypeUMAP,
    plotCmpPerCellType,
    plotCmpUMAP,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..imports.gating import gateThomsonCells
import numpy as np
import umap 
import umap.plot
import seaborn as sns


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (3, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    dataDF = flattenData(data)

    rank = 30
    
    dataDF["Cell Type"] = gateThomsonCells(rank=rank, saveCellTypes=False)

    _, factors, projs = openPf2(rank, "Thomson")
    pf2Points = openUMAP(rank, "Thomson", opt=False)
    
    weightedProjDF = flattenWeightedProjs(data, factors, projs)
    weightedProjDF["Cell Type"] = dataDF["Cell Type"].values
    weightedProjDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    dataDF.sort_values(by=["Condition", "Cell Type"], inplace=True)

    cmp=20
    comps = [cmp]
    for i, comp in enumerate(comps):
        # plotCmpPerCellType(weightedProjDF, comps[i], ax[(2 * i) +1], outliers=False)
        # plotCmpUMAP(comps[i], factors, pf2Points, projs, ax[(2 * i) + 2])
        b = plotCmpUMAP(comps[i], factors, pf2Points, projs, ax[i])

    
    pf2Points = umap.UMAP(random_state=1).fit_transform(projs)
    
    cellSkip = 10
    umap1 = pf2Points[::cellSkip, 0]
    umap2 = pf2Points[::cellSkip, 1]
    weightedProjs = projs @ factors[1]
    a = weightedProjs[:, cmp-1]
    weightedProjs = weightedProjs[:, cmp-1]

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    weightedProjs = weightedProjs[::cellSkip]
    weightedProjs = weightedProjs / np.max(np.abs(weightedProjs))
    psm = plt.pcolormesh([[-1, 1],[-1, 1]], cmap=cmap)

    ax[1].scatter(
            umap1,
            umap2,
            c=weightedProjs,
            cmap=cmap,
            s=0.2,
            alpha=1
        )
    plt.colorbar(psm, ax=ax[1])

    # plot = umap.plot.points(pf2Points, values=weightedProjs, cmap=cmap, subset_points= subset, ax=ax)
    # colorbar= plt.colorbar(psm, ax=plot)
    ax[1].set(
        ylabel="UMAP2",
        xlabel="UMAP1",
        title="Component:" + str(cmp),
        xticks=np.linspace(np.min(umap1), np.max(umap1), num=5),
        yticks=np.linspace(np.min(umap2), np.max(umap2), num=5),
    )

    ax[1].axes.xaxis.set_ticklabels([])
    ax[1].axes.yaxis.set_ticklabels([])
 
    # subset = np.random.choice(a=[False, True], size=len(dataDF["Cell Type"].values), p=[.75, .25])
    # umap.plot.points(pf2Points,subset_points=subset, ax=ax[0])
    # ax[0].set(
    #     ylabel="UMAP2",
    #     xlabel="UMAP1")

    
        # )

    # plotFactors(factors, data, ax[0:3], reorder=(0, 2), trim=(2,), saveGenes=False)

    # pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))
    # dataDF = flattenData(data)
    # cond = ["control", "sc_pod1", "sc_pod7", "ic_pod1", "ic_pod7"]
    # plotCondUMAP(cond, "Pf2", dataDF["Condition"].values, pf2Points, ax[3:9])
    
    

    # # plotCellTypeUMAP(pf2Points, dataDF, ax[0])


    np.testing.assert_allclose(a,b)


    # geneSet1 = ["NKG7", "GNLY", "GZMB", "GZMH", "PRF1"]
    # geneSet2 = ["MS4A1", "CD79A", "CD79B", "TNFRSF13B", "BANK1"]

    # # genes = [geneSet1, geneSet2]
    # # for i in range(len(genes)):
    # #     plotGenePerCellType(genes[i], dataDF, ax[i + 9])

    # # set3 = ["VPREB3", "CD79A", "FAM111B", "HOPX", "SLC30A3", "MS4A1"]
    # # set4 = ["CD163", "ADORA3", "MS4A6A", "RNASE1", "MTMR11"]

    # glucs = [
    #     "Betamethasone Valerate",
    #     "Loteprednol etabonate",
    #     "Budesonide",
    #     "Triamcinolone Acetonide",
    #     "Meprednisone",
    # ]
    # geneSet3 = ["CD163", "ADORA3", "MS4A6A", "RNASE1", "MTMR1", "NDRG2", "CRABP2", "AK8"]
    # geneSet4 = ["CCNB1", "GADD45A", "SLC40A1", "CDC20", "ITIH3", "VPREB3", "CD79A", "FAM111B", "HOPX", "SLC30A3"]
    
    
    # geneSet4 = ["CCNB1", "GADD45A", "SLC40A1", "CDC20", "ITIH3", "CPEB1", "CBR3", "MT1H", "MT1G", "PLK1", "FBN1", "CCR7", "TPX2", 
    #             "FAM131C", "MT1M", "PRSS57", "CENPE", "LTA"]
    
    # geneSet4 = ["VPREB3", "CD79A", "FAM111B", "HOPX", "SLC30A3", "MS4A1", "TYMA", "RRM2", "TESC", "CD70", "BANK1", "CD79B",
    # "KIAA0101", "ZBED2", "CTSW", "CXCL9", "CCNA2", "CXCL10", "MKI67"]

    # plotGenePerCategCond(glucs, "Gluco", geneSet3, dataDF, ax[6:14])

    # geneSet4 = ["VPREB3", "FAM111B"]
    # plotGenePerCategCond(
    #     ["Dexrazoxane HCl (ICRF-187, ADR-529)"], "Dex HCl", geneSet4, dataDF, ax[0:20]
    # )
    


    
    
    
    


    return f
