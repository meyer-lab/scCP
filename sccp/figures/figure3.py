"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
from .common import (
    subplotLabel,
    getSetup,
    flattenData,
    plotDrugUMAP,
    plotGeneUMAP,
    plotCmpUMAP,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import umap
from sklearn.decomposition import PCA


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 10), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes(offset=1.0)
    rank = 25
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
        verbose=True,
    )

    dataDF, _, _ = flattenData(data, factors, projs)

    # UMAP dimension reduction
    umapReduc = umap.UMAP(random_state=1)
    pf2Points = umapReduc.fit_transform(np.concatenate(projs, axis=0))

    # PCA dimension reduction
    pc = PCA(n_components=rank)
    pcaPoints = pc.fit_transform(data.unfold())
    pcaPoints = umapReduc.fit_transform(pcaPoints)

    # NK, CD4, B, CD8
    
#     MS4A1, CD79A
# Natural killer (NK) cells	GNLY, NKG7, KLRB1
# Dendritic Cells	FCER1A, CST3
# Megakaryocytes	PPBP
# FCGR3A+ Monocytes	FCGR3A, MS4A7

    genes = ["NKG7", "GNLY", "CD8A", "IL7R", "SELL", "MS4A1", "FCGR3A", "MS4A7", "CD14"]
    plotGeneUMAP(genes, "Pf2", pf2Points, dataDF, f, ax[0:9])
    # plotGeneUMAP(genes, "PCA", pcaPoints, dataDF, f, ax[1:2])

    # # Find cells associated with drugs
    # drugs = [
    #     "Triamcinolone Acetonide"]
    #     # "Alprostadil",
    # # ]
    # plotDrugUMAP(drugs, "Pf2", dataDF["Drug"].values, pf2Points, ax[2:3])
    # plotDrugUMAP(drugs, "PCA", dataDF["Drug"].values, pcaPoints, ax[3:4])
    
    return f
