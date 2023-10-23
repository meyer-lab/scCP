"""
Lupus: UMAP only progen cell 
"""
from .common import getSetup, openPf2
from .commonFuncs.plotUMAP import plotCmpPerCellType, points, plotCmpUMAP, plotAllLabelsUMAP
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.axes import Axes
from pacmap import PaCMAP



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 5), (2, 2))

    X = openPf2(rank=40, dataName="Lupus")

    # X = X[X.obs["Cell Type"].isin(["Progen"])]
    # X = X[X.obs["SLE_status"].isin(["SLE"])]
    # print(X.obs["louvain"])
    # X.obs["louvain"] = np.where(X.obs["louvain"].isin(["14"]), X.obs["louvain"], "Other")
    
    X = X[X.obs["louvain"].isin(["14"])]
    cmp=13
    
    print(X.obsm["weighted_projections"][:, cmp])
    
    X = X[X.obsm["weighted_projections"][:, cmp] > .1, :] 
    print(X)
    
    # c
    
    # cmp = 13
    # X.obsm["weighted_projections"][:, cmp] > 0.1, :
    
    # X = X[X.obs['].X > 0.5, :] 
    # adata = adata[adata[: , 'elav'].X > 0.5, :] 
    
    # print(X.obs["Cell Type"])
    # print(X.obs["Cell Type"].count())
    
   
    
    # cmp = 13
    # X.obsm["embedding"] = PaCMAP(random_state=1).fit_transform(X.obsm["projections"])
    # plotPartialLabelUMAP(X, ax=ax[2])
    # plotPartialCmpUMAP(X, cmp, ax=ax[0])
    #  plotPartialLabelUMAP(X, ax: Axes):
    # plotPartialLabelUMAP(X, ax=ax[2])
    plotCmpUMAP(X, cmp=cmp, ax=ax[0])
    # plotAllLabelsUMAP(X, "SLE_status", ax[2])
    
    # print(X.obsm["weighted_projections"][:, cmp - 1])
    
    # weightedProjs = X.obsm["weighted_projections"]
    # weightedProjs = weightedProjs[:, cmp - 1]
    # weightedProjs = weightedProjs / np.max(np.abs(weightedProjs))
    
    # sns.histplot(weightedProjs, bins=20, ax=ax[1])


    return f


def plotPartialCmpUMAP(X, cmp: int, ax: Axes):
    """Scatterplot of UMAP visualization weighted by projections for a component"""
    weightedProjs = X.obsm["weighted_projections"]
    weightedProjs = weightedProjs[:, cmp - 1]
    weightedProjs = weightedProjs / np.max(np.abs(weightedProjs))
    weightedProjs[0] = -1
    weightedProjs[1] = 1
    
    cmap = sns.diverging_palette(240, 10, as_cmap=True, s=100)

    ax.scatter(
            X.obsm["embedding"][:, 0],
            X.obsm["embedding"][:, 1],
            c=weightedProjs,
            cmap=cmap,
            s=0.5,
        )
    psm = plt.pcolormesh([[-1, 1], [-1, 1]], cmap=cmap)
    ax.set(ylabel="UMAP2", xlabel="UMAP1", title="Cmp. " + str(cmp),
        xticks=np.linspace(np.min(X.obsm["embedding"][:, 0]), np.max(X.obsm["embedding"][:, 0]), num=5),
        yticks=np.linspace(np.min(X.obsm["embedding"][:, 1]), np.max(X.obsm["embedding"][:, 1]), num=5))
    plt.colorbar(psm, ax=ax, label="Cell Specific Weight")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])


def plotPartialLabelUMAP(X, ax: Axes):
    sns.scatterplot(x=X.obsm["embedding"][:, 0], y=X.obsm["embedding"][:, 1], hue=X.obs["SLE_status"], s=5, palette="muted", ax=ax)
    ax.set(ylabel="UMAP2", xlabel="UMAP1", 
        xticks=np.linspace(np.min(X.obsm["embedding"][:, 0]), np.max(X.obsm["embedding"][:, 0]), num=5),
        yticks=np.linspace(np.min(X.obsm["embedding"][:, 1]), np.max(X.obsm["embedding"][:, 1]), num=5))
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])



# def plotCond2UMAP(decompType: str, XX,  ax: Axes):
#     """Scatterplot of UMAP visualization weighted by condition"""
#     X = XX.copy()
#     X.obs["Condition"] = np.array(
#         [c if c in condition else " Other Conditions" for c in X.obs["Condition"]]
#     )
#     points(
#         X.obsm["embedding"],
#         labels=X.obs["SLE_status"],
#         ax=ax,
#         color_key_cmap="tab20",
#         show_legend=True,
#     )
#     ax.set(title=f"{decompType}-Based Decomposition", ylabel="UMAP2", xlabel="UMAP1")
