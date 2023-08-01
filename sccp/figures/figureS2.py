"""
S2: Examining cell state for Pf2 on lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: visualize the cell state compostition by cell type/UMAP

from .common import (
    subplotLabel,
    getSetup,
    plotCmpUMAP,
    plotCompViolins,
    plotUMAP_ct,
    openPf2
)
from ..imports.scRNA import load_lupus_data
import umap
from sklearn.decomposition import PCA

def plotUMAP_obslabel(labels, pf2Points, ax):
    """Scatterplot of UMAP visualization labeled by cell type"""
    plot = umap.plot.points(pf2Points, 
                            labels = labels, 
                            theme='viridis', 
                            ax=ax)
    ax.set(
        ylabel="UMAP2",
        xlabel="UMAP1",
        title="Pf2-Based Decomposition: Label Cell Types")

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of data
    data, louvain, _ = load_lupus_data(obs_return='louvain')
    _, status, _ = load_lupus_data(obs_return='Status')
    _, age, _ = load_lupus_data(obs_return='Age')
    rank = 39
    cellState = 16
    cmp = 16

    _, factors, projs, = openPf2(rank = rank, dataName = 'lupus', optProjs=True)


   # proj_B = projs @ factors[1]

    # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1, verbose=True).fit(projs)


    plotCmpUMAP(cellState, cmp, factors, pf2Points, projs, ax[0])
    plotUMAP_obslabel(louvain, pf2Points, ax[1])
    plotUMAP_obslabel(status, pf2Points, ax[2])
    plotUMAP_obslabel(age, pf2Points, ax[3])

    return f
