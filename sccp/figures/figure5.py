"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
from .common import (
    subplotLabel,
    getSetup,
    flattenData,
    plotDrugUMAP,
    plotGeneUMAP
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import umap 



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes(offset=1.0)

    _, factors, projs, _ = parafac2_nd(
        data,
        rank=30,
    )

    dataDF, projDF = flattenData(data, factors, projs)

    # UMAP dimension reduction
    cmpNames = [f"Cmp. {i}" for i in np.arange(1, factors[0].shape[1] + 1)]
    umapReduc = umap.UMAP()
    umapPoints = umapReduc.fit_transform(projDF[cmpNames].to_numpy())
     
    # Mono1, Mono2, NK, DC, CD8, CD4, B
    genes = ["CD14", "FCGR3A", "NKG7", "CST3", "CD8B", "IL7R", "MS4A1"]
    plotGeneUMAP(genes, umapPoints, dataDF, ax[0:7])

    # Find cells associated with drugs
    drugs = ["CTRL2", "Triamcinolone Acetonide", "Budesonide", "Betamethasone Valerate", "Dexrazoxane HCl (ICRF-187, ADR-529)"]
    plotDrugUMAP(drugs, dataDF["Drug"].values, umapPoints, ax[7:12])

    return f

 