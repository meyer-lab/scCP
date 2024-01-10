"""
Thomson: XX
"""
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotUMAP import plotLabelsUMAP, plotCmpUMAP, plotGeneUMAP
from .commonFuncs.plotGeneral import (
    plotGenePerCellType,
    plotGenePerCategStatus,
    plot2GenePerCategStatus,
    plotGeneFactors,
    gene_plot_cells,
    plot_cell_gene_corr,
    heatmapGeneFactors,
)
import scanpy as sc
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.axes import Axes
from pacmap import PaCMAP
import scanpy as sc
from ..imports import import_lupus

# def makeFigure():
#     """Get a list of the axis objects and create a figure."""
#     # Get list of axis objects
#     ax, f = getSetup((15, 18), (4, 3))

#     # Add subplot labels
#     subplotLabel(ax)


#     genes = ["CDKN1A", "APOBEC3A", "LPAR6", "DPYSL2", "ABI3", "PLBD1", "RBP7", "S100A12", "CDA", "S100A8"] #43
#     # genes = ["SGK1", "MAFB", "DUSP6", "C5AR1", "RGCC", "CLEC4E", "IRS2", "LRRK2", "IER3", "PDE4B"] #22
#     # genes = ["RGS1", "PTGER4", "PMAIP1", "NR4A2", "CD83", "CLDND1", "AC092580.4", "BANK1", "PTGER2", "ESF1"] #30
#     # genes = ["IL8", "MGST1", "IRS2", "NR4A2", "S100A12", "CDA", "APOBEC3A", "ISG15", "IFITM3", "MX1"] #39
#     # genes = ["IFI27", "IFITM3", "IFI6", "ISG15", "APOBEC3A", "RETN", "CD8A", "PTGER4", "ESF1", "ANXA2R"] # 48
#     X = X.to_memory()
#     # ind = X.obs["Cell Type"] == ("cM" or "ncM")
#     # X = X[ind, :]
#     # ind
    
    
    
#     cmp = 48
#     ind = X.obsm["weighted_projections"] > .1
#     X = X[ind[:, cmp-1], :]
   
 


#     # ind = X.obsm["Cell"] < -.1
#     # X = X[ind[:, cmp-1], :]
   
#     # # X = sc.pp.subsample(X, fraction=0.01, random_state=0, copy=True)  
#     genes = ["IFI27", "RETN"] 
#     plot2GenePerCategStatus(["SLE"], "lupus", genes[0],genes[1], X, ax[0], obs = "SLE_status", mean=True, cellType="Cell Type")

#     for i, gene in enumerate(genes):
#         plotGenePerCategStatus(["SLE"], "lupus", gene, X, ax[i+1], obs = "SLE_status", mean=True, cellType="Cell Type")
        




#     return f

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 18), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    XX = read_h5ad("factor_cache/Lupus.h5ad", backed="r")
    plotLabelsUMAP(XX, "Cell Type", ax[0])
    
    cmp = 43
    ind = XX.obsm["weighted_projections"] < -.08
    XXX = XX[ind[:, cmp-1], :]
    plotPartialCmpUMAP(XXX, 43, ax=ax[1])
    
    

    plotPartialLabelUMAP(XXX, ax[2], obslabel="Cell Type")
    plotPartialLabelUMAP(XXX, ax[3], obslabel="louvain")
    plotPartialLabelUMAP(XXX, ax[4], obslabel="SLE_status")

    


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
            X.obsm["X_pf2_PaCMAP"][:, 0],
            X.obsm["X_pf2_PaCMAP"][:, 1],
            c=weightedProjs,
            cmap=cmap,
            s=0.5,
        )
    psm = plt.pcolormesh([[-1, 1], [-1, 1]], cmap=cmap)
    ax.set(ylabel="UMAP2", xlabel="UMAP1", title="Cmp. " + str(cmp),
        xticks=np.linspace(np.min(X.obsm["X_pf2_PaCMAP"][:, 0]), np.max(X.obsm["X_pf2_PaCMAP"][:, 0]), num=5),
        yticks=np.linspace(np.min(X.obsm["X_pf2_PaCMAP"][:, 1]), np.max(X.obsm["X_pf2_PaCMAP"][:, 1]), num=5))
    plt.colorbar(psm, ax=ax, label="Cell Specific Weight")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

def plotPartialLabelUMAP(X, ax: Axes, obslabel:str):
    sns.scatterplot(x=X.obsm["X_pf2_PaCMAP"][:, 0], y=X.obsm["X_pf2_PaCMAP"][:, 1], hue=X.obs[obslabel], s=5, palette="muted", ax=ax)
    ax.set(ylabel="UMAP2", xlabel="UMAP1", 
        xticks=np.linspace(np.min(X.obsm["X_pf2_PaCMAP"][:, 0]), np.max(X.obsm["X_pf2_PaCMAP"][:, 0]), num=5),
        yticks=np.linspace(np.min(X.obsm["X_pf2_PaCMAP"][:, 1]), np.max(X.obsm["X_pf2_PaCMAP"][:, 1]), num=5))
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])