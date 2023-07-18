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
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes(offset=1.0)
    rank = 30
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
        verbose=True,
    )

    dataDF = flattenData(data)

    # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))

    # # PCA dimension reduction
    # pc = PCA(n_components=rank)
    # pcaPoints = pc.fit_transform(data.unfold())
    # pcaPoints = umap.UMAP(random_state=1).fit(pcaPoints)
    cd4 =  ["IL7R", "CCR7"]
    cd8 =  ["CD8A", "CD8B"] 
    nk =  ["GNLY", "NKG7", "KLRB1"] 
    mono1 =   ["CD14", "LYZ", "MS4A7"]
    mono2 = ["FCGR3A", "CST3"] 
    megakary = ["PPBP"]
    b = ["MS4A1", "CD79A"]
    
    plotGeneUMAP(np.concatenate((cd4, cd8, nk, mono1, mono2, megakary, b)), "Pf2", pf2Points, dataDF, ax[0:16])
    # plotGeneUMAP(genes, "PCA", pcaPoints, dataDF, ax[2:4])

    # Find cells associated with drugs
    # drugs = [
    #     "Triamcinolone Acetonide",
    #     "Alprostadil",
    # ]
    # plotDrugUMAP(drugs, "Pf2", dataDF["Condition"].values, pf2Points, ax[4:6])
    # plotDrugUMAP(drugs, "PCA", dataDF["Condition"].values, pcaPoints, ax[6:8])
    
    return f
