"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
from .common import (
    subplotLabel,
    getSetup,
    flattenData,
    plotDrugDimReduc,
    plotGeneDimReduc
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import umap 
from sklearn.decomposition import PCA


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes(offset=1.0)

    rank = 30
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
    )

    dataDF, projDF = flattenData(data, factors, projs)

    # UMAP dimension reduction
    cmpNames = [f"Cmp. {i}" for i in np.arange(1, factors[0].shape[1] + 1)]
    umapReduc = umap.UMAP()
    pf2Points = umapReduc.fit_transform(projDF[cmpNames].to_numpy())
    
    # PCA dimension reduction
    pc = PCA(n_components=rank)
    pcaPoints = pc.fit_transform(dataDF[data.variable_labels].to_numpy())
    pcaPoints = umapReduc.fit_transform(pcaPoints)
     
    # Mono1, Mono2, NK, CD4, B
    genes = ["CD14", "FCGR3A", "NKG7", "IL7R", "MS4A1"]
    plotGeneDimReduc(genes, ["UMAP1", "UMAP2"], pf2Points, dataDF, ax[0:5])
    plotGeneDimReduc(genes, ["PCA1", "PCA2"], pcaPoints, dataDF, ax[5:10])

    # Find cells associated with drugs
    drugs = ["Triamcinolone Acetonide", "Budesonide", "Betamethasone Valerate"]
    plotDrugDimReduc(drugs, ["UMAP1", "UMAP2"], dataDF["Drug"].values, pf2Points, ax[10:13])
    plotDrugDimReduc(drugs, ["PCA1", "PCA2"], dataDF["Drug"].values, pcaPoints, ax[13:16])
    
    
    return f

 