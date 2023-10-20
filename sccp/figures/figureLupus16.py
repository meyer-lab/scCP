"""
Lupus: UMAP only progen cell 
"""
from .common import getSetup, openPf2
from .commonFuncs.plotUMAP import plotCmpPerCellType, points
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.axes import Axes
from pacmap import PaCMAP



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 2))

    X = openPf2(rank=40, dataName="Lupus")

    X = X[X.obs["Cell Type"].isin(["Progen"])]
    
    cmp = 13
    X.obsm["embedding"] = PaCMAP(random_state=1).fit_transform(X.obsm["projections"])
    plotPartialCmpUMAP(X, cmp=cmp, ax=ax[0])
    
    print(X.obsm["weighted_projections"][:, cmp - 1])
    
    sns.histplot(X.obsm["weighted_projections"][:, cmp - 1], bins=20, ax=ax[1])

    return f


def plotPartialCmpUMAP(X, cmp: int, ax: Axes):
    """Scatterplot of UMAP visualization weighted by
    projections for a component and cell state"""
    weightedProjs = X.obsm["weighted_projections"]
    weightedProjs = weightedProjs[:, cmp - 1]
    weightedProjs = weightedProjs / np.max(np.abs(weightedProjs)) * 2.0

    cmap = sns.diverging_palette(240, 10, as_cmap=True, s=100)
    plot = points(X.obsm["embedding"], values=weightedProjs, cmap=cmap, ax=ax)

    psm = plt.pcolormesh([[-1, 1], [-1, 1]], cmap=cmap)
    plt.colorbar(psm, ax=plot, label="Cell Specific Weight")
    ax.set(ylabel="UMAP2", xlabel="UMAP1", title="Cmp. " + str(cmp))
