"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
from sklearn.decomposition import PCA
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotUMAP import plotGeneUMAP, points, plotCondUMAP
from ..imports.scRNA import import_thomson
import pacmap




def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    rank = 30
    X = openPf2(rank, dataName="Thomson")
    
    genes = ["GNLY", "NKG7"]
    gene = "NKG7"
    plotGeneUMAP(gene, "Pf2", X, ax[0])
    
    
    drugs = ["Triamcinolone Acetonide", "Alprostadil"]
    drug = "Triamcinolone Acetonide"
    
    
    # PCA dimension reduction
    pc = PCA(n_components=rank)
    pcaPoints = pc.fit_transform(X.X)
    X.obsm["embedding"] = pacmap.PaCMAP().fit_transform(pcaPoints)
    
    plotCondUMAP(drug, "Pf2", X, ax[0])
    

    # points(
    #     pf2Points,
    #     labels=condList,
    #     ax=ax[4],
    #     color_key_cmap="Paired",
    #     show_legend=True,
    # )
    # ax[4].set(
    #     title="Pf2-Based Decomposition", ylabel="UMAP2", xlabel="UMAP1"
    # )






    # plotGeneUMAP(genes, "PCA", pcaPoints, dataDF, ax[2:4])

    # # Find cells associated with drugs
    # drugs = [
    #     "Triamcinolone Acetonide",
    #     "Alprostadil",
    # ]
    # condList = np.array([c if c in drugs else " Other Conditions" for c in dataDF["Condition"].values])

    # points(
    #     pf2Points,
    #     labels=condList,
    #     ax=ax[4],
    #     color_key_cmap="Paired",
    #     show_legend=True,
    # )
    # ax[4].set(
    #     title="Pf2-Based Decomposition", ylabel="UMAP2", xlabel="UMAP1"
    # )

    # points(
    #     pcaPoints,
    #     labels=condList,
    #     ax=ax[5],
    #     color_key_cmap="Paired",
    #     show_legend=True,
    # )
    # ax[5].set(
    #     title="PCA-Based Decomposition", ylabel="UMAP2", xlabel="UMAP1"
    # )

    return f
