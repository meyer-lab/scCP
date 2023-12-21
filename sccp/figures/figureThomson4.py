"""
Thomson: Compares PCA and Pf2 UMAP labeled by genes and drugs
"""
import anndata
from sklearn.decomposition import PCA
from .common import subplotLabel, getSetup
from .commonFuncs.plotUMAP import plotGeneUMAP, plotLabelsUMAP
import pacmap
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((50, 50), (7, 7))

    # Add subplot labels
    subplotLabel(ax)

    X = anndata.read_h5ad("factor_cache/Thomson.h5ad", backed="r")
    
    drugs = np.unique(X.obs["Condition"])
    # for i, drug in enumerate(drugs):
    #     print(drug)
    #     plotLabelsUMAP(X, "Condition", ax[i], drug, cmap="Set1")
    #     ax[i].set(title=f"Pf2 Embedding")
      
        

    # PCA dimension reduction
    pc = PCA(n_components=30)
    pcaPoints = pc.fit_transform(np.asarray(X.X.to_memory() - X.var["means"].values))
    X.obsm["embedding"] = pacmap.PaCMAP().fit_transform(pcaPoints)

    for i, drug in enumerate(drugs):
        print(drug)
        plotLabelsUMAP(X, "Condition", ax[i], drug, cmap="Set1")
        ax[i].set(title=f"Standard Embedding")

    return f
