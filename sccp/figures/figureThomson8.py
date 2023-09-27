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
from .commonFuncs.plotGeneral import plotGenePerCellType, plotGenePerCategCond
from .commonFuncs.plotUMAP import (
    plotCellTypeUMAP,
    plotCmpPerCellType,
    plotCmpUMAP,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..imports.gating import gateThomsonCells

import datashader as ds
import datashader.transfer_functions as tf
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (2, 1))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    dataDF = flattenData(data)

    rank = 30
    dataDF["Cell Type"] = gateThomsonCells(rank=rank, saveCellTypes=False)

    _, factors, projs = openPf2(rank, "Thomson")
    pf2Points = openUMAP(rank, "Thomson", opt=False)

    # plotCellTypeUMAP(pf2Points, dataDF, ax[0])

    # weightedProjDF = flattenWeightedProjs(data, factors, projs)
    # weightedProjDF["Cell Type"] = dataDF["Cell Type"].values
    # weightedProjDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    # dataDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    
    pf2Points = umap.UMAP(random_state=1).fit_transform(projs)
    
    print(np.shape(pf2Points))
    print(np.shape(projs))
    df = pd.DataFrame(data=pf2Points, columns=["UMAP1", "UMAP2"])
    # df["Weight"] = projs[:, 0]
    values = projs[:, 0]
    cvs = ds.Canvas()
    # agg = cvs.points(df, "UMAP1", "UMAP2")
    cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
    
    def _to_hex(arr):
        return [matplotlib.colors.to_hex(c) for c in arr]
    
    
    min_val, max_val = np.min(values), np.max(values)
    bin_size = (max_val - min_val) / 255.0
    df["val_cat"] = pd.Categorical(np.round((values - min_val) / bin_size).astype(np.float32))
    aggregation = cvs.points(df, "UMAP1", "UMAP2", agg=ds.count_cat("val_cat"))
    color_keys = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, 256)))
    img = tf.shade(aggregation, color_key=color_keys)
    img = tf.set_background(img, "white")
    
    # data["val_cat"] = pd.Categorical(projs[:, 0])
    # aggregation = cvs.points(data, "x", "y", agg=ds.count_cat("val_cat"))
    #         color_key_cols = _to_hex(
    #             plt.get_cmap(cmap)(np.linspace(0, 1, unique_values.shape[0]))
    #         )
    #         color_key = dict(zip(unique_values, color_key_cols))
    #         result = tf.shade(
    #             aggregation, color_key=color_key, how="eq_hist", alpha=alpha
    
    # img = tf.shade(agg, cmap="lightblue")
    
    plt.imshow(img)
    plt.setp(ax[0])
    ax[0].set_ylim(ymin=0)
    
    # tf.shade(ds.Canvas().points(df,"UMAP1", "UMAP2"), ax=ax[0])
    
    

    # comps = [5, 12, 20, 30]
    # for i, comp in enumerate(comps):
    #     # plotCmpPerCellType(weightedProjDF, comps[i], ax[(2 * i) + 1], outliers=False)
    #     plotCmpUMAP(comps[i], factors, pf2Points, projs, ax[i])
        

    # geneSet1 = ["NKG7", "GNLY", "GZMB", "GZMH", "PRF1"]
    # geneSet2 = ["MS4A1", "CD79A", "CD79B", "TNFRSF13B", "BANK1"]

    # genes = [geneSet1, geneSet2]
    # for i in range(len(genes)):
    #     plotGenePerCellType(genes[i], dataDF, ax[i + 9])

    # # set3 = ["VPREB3", "CD79A", "FAM111B", "HOPX", "SLC30A3", "MS4A1"]
    # # set4 = ["CD163", "ADORA3", "MS4A6A", "RNASE1", "MTMR11"]

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
