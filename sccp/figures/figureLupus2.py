"""
Lupus: UMAP labeled by cell type
"""
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotUMAP import plotLabelsUMAP
from .commonFuncs.plotGeneral import plotGenePerCategCond
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.axes import Axes
from pacmap import PaCMAP
import scanpy as sc
from ..imports import import_lupus

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 10), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("factor_cache/Lupus.h5ad", backed="r")

    # X = import_lupus()
    cmp = 15
    # plotLabelsUMAP(X, "Cell Type", ax[0])

    # # ind = X.obsm["weighted_projections"] < -.03
    # # X = X[ind[:, cmp-1], :]

    # # # X = sc.pp.subsample(X, fraction=0.1, random_state=0, copy=True)
    # # # plotPartialCmpUMAP(X, cmp, ax[0])
    # # # print(X)
    # # # plotPartialLabelUMAP(X, ax[0], obslabel="Cell Type")
    # # # plotPartialLabelUMAP(X, ax[1], obslabel="louvain")
    # # # plotPartialLabelUMAP(X, ax[2], obslabel="SLE_status")
    
    genes = ["TUBB1", "PF4", "CLU", "PPBP", "HIST1H2AC", "ISG15", "LY6E", "IFI44L", "ANXA1", "IFI6"]
    
    genes = ["KLRB1", "GZMK", "KLRG1"]
             
    for i, gene in enumerate(genes):
        plotGenePerCategCond(["SLE"], "lupus", gene, X, ax[i], obs = "SLE_status", mean=True, cellType="Cell Type", raw=False)
        print(i)
    # dfjkda;ls
    
    
    # print(XX)



    return f


def plotPartialCmpUMAP(X, cmp: int, ax):
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
    
def plotPartialLabelUMAP(X, ax: Axes, obslabel:str):
    sns.scatterplot(x=X.obsm["embedding"][:, 0], y=X.obsm["embedding"][:, 1], hue=X.obs[obslabel], s=5, palette="muted", ax=ax)
    ax.set(ylabel="UMAP2", xlabel="UMAP1", 
        xticks=np.linspace(np.min(X.obsm["embedding"][:, 0]), np.max(X.obsm["embedding"][:, 0]), num=5),
        yticks=np.linspace(np.min(X.obsm["embedding"][:, 1]), np.max(X.obsm["embedding"][:, 1]), num=5))
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])