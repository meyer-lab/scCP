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
import umap.plot
from sklearn.decomposition import PCA


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes(offset=1.0)
    rank = 25
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
        verbose=True,
    )

    dataDF, projDF, _ = flattenData(data, factors, projs)
    
        #  flowDF = flowDF[
        #         flowDF[mark] < flowDF[mark].quantile(0.995)
    
    weight = 0
    for i, celltypes in enumerate(canonicalGenes):
        genez = np.asarray(canonicalGenes[celltypes])
        print(celltypes)
        print(genez)
        
        if len(genez) == 1:
            dataDF.loc[(dataDF[genez[0]] > weight), "Cell Type"] = celltypes
        elif len(genez) == 2:
            dataDF.loc[(dataDF[genez[0]] > weight) & (dataDF[genez[1]] > weight), 
                       "Cell Type"] = celltypes  
        elif len(genez) == 3:
            dataDF.loc[(dataDF[genez[0]] > weight) & (dataDF[genez[1]] > weight)
                       & (dataDF[genez[2]] > weight), "Cell Type"] = celltypes  
            
    print(dataDF)
    
    dataDF["Cell Type"] = dataDF["Cell Type"].fillna("Other")
    
    print(dataDF)
        
    # dataDF = dataDF.dropna()
    # idx = dataDF.index.values
    
    # print(dataDF)
        # print(dataDF)

    # for marker in canonicalGenes:
    #     dataDF.loc[dataDF[marker] > 0.03, "Cell Type"] = "Marker Positive"
    
    # proj = np.concatenate(projs, axis=0)

    # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))

    # # PCA dimension reduction
    # pc = PCA(n_components=rank)
    # pcaPoints = pc.fit_transform(data.unfold())
    # pcaPoints = umap.UMAP(random_state=1).fit(pcaPoints)

    # genes = ["GNLY", "NKG7"]
    # plotGeneUMAP(genes, "Pf2", pf2Points, dataDF, ax[0:2])
    # plotGeneUMAP(genes, "PCA", pcaPoints, dataDF, ax[2:4])

    # # Find cells associated with drugs
    # drugs = [
    #     "Triamcinolone Acetonide",
    #     "Alprostadil",
    # ]
    # plotDrugUMAP(drugs, "Pf2", dataDF["Drug"].values, pf2Points, ax[4:6])
    # plotDrugUMAP(drugs, "PCA", dataDF["Drug"].values, pcaPoints, ax[6:8])
    
    subset = np.random.choice(a=[False, True], size=len(dataDF["Cell Type"].values), p=[.93, .07])
    
    umap.plot.points(pf2Points, labels=dataDF["Cell Type"].values, ax=ax[0], color_key_cmap="tab20", show_legend=True, subset_points=subset)
        # axs[i].set(
        #     title=decomp + "-Based Decomposition",
        # ylabel="UMAP2",
        # xlabel="UMAP1")
    
    # cmp = [1, 25]
    # allP = np.concatenate(projs, axis=0)
    # CompW = allP @ factors[1]
    # plotCmpUMAP(CompW, cmp, pf2Points, ax[8:10])
    
    
    
    return f


""""""

canonicalGenes = {
    "CD4 Memory T ": ["CD3D", "CREM"],
    # "CD4 Native T ": ["CD3D", "SELL", "GIMAP5"],
    # "Activated T ": ["CREM", "CACYBP"],
    "Natural Killer": ["GNLY", "NKG7"],
    "CD8 T": ["CD3D", "NKG7", "CD8A"],
    "B": ["CD79A", "MS4A1"],
    "CD16 Monocytes": ["FCGR3A", "VMO1"],
    "CD14 Monocytes": ["CD14", "CCL2", "S100A9"],
    "Dentritic Cells": ["HLA-DQA1", "GPR183"],
    "Megakaryocyte": ["PPBP", "GNG11"],
    "Plasmacytoid Dendritic Cells": ["TSPAN13", "IGJ"]}

canonicalGenes2 = {
    # "CD4 Native T ": ["CD3D", "SELL", "GIMAP5"],
    # "Activated T ": ["CREM", "CACYBP"],
    "Natural Killer": ["GNLY", "NKG7"],
    "CD8 T": ["CD8A"],
    "B": ["MS4A1"],
    "FCGR3A Monocytes": ["FCGR3A", "MS4A7"],
    "CD14 Monocytes": ["CD14", "LYZ", "S100A8"],
    "Dentritic Cells": ["GPR183", "HLA-DQA1"],
    "Megakaryocyte": ["PPBP"],
     "CD4 T ": ["CD3D", "CREM"],}
    # "Plasmacytoid Dendritic Cells": ["TSPAN13", "IGJ"]}


# def distAllGeneDF(data, Pf2s, PCs):
#     """Plots normalized variance for either a variable or for a group of cells"""
#     distDF = pd.DataFrame([])
    
#     factors = Pf2s[1]
#     projs = Pf2s[2]
#     dataDF, projDF, _ = flattenData(data, factors, projs)
#     pf2All = projDF.values[:, 0:-1]
#     pcaAll = PCs
    
#     markers = [item for value in marker_genes.values() for item in (value if isinstance(value, list) else [value])]

#     datDFcopy = dataDF.copy()
#     for marker in markers:
#         if marker in datDFcopy.columns:
#             dataDF.loc[dataDF[marker] > 0.03, marker + " status"] = "Marker Positive"
            
#     for marker in markers:
#         if marker in datDFcopy.columns:
#             pf2Gene = projDF.loc[dataDF[marker + " status"] == "Marker Positive"].values[:, 0: -1]
#             pcaGene = pcaAll[dataDF[marker + " status"] == "Marker Positive"]

#             pf2Dist = centroid_dist(pf2Gene) / centroid_dist(pf2All)
#             pcaDist = centroid_dist(pcaGene) / centroid_dist(pcaAll)