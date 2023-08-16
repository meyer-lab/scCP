"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup, plotCmpUMAP, openPf2, plotCmpUMAP2, saveUMAP
from ..imports.scRNA import ThompsonXA_SCGenes
from parafac2 import parafac2_nd
import umap 
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    # ax, f = getSetup((25, 25), (4,4))
    ax, f = getSetup((14, 12), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    rank = 30


    weight, factors, projs = openPf2(rank, "Thomson")
    
    # saveUMAP(fit_points, rank:int, dataName: str):
    
    
    
    # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1).fit(projs)
    # saveUMAP(pf2Points, rank=30, dataName="Thomoson")
    
    # component = np.arange(17, 25, 1) 
    
    component = [5, 30]
    
    
    
    
    for i in range(len(component)):
        # plotCmpUMAP(component[i], factors, pf2Points, projs, ax[2*i])
        plotCmpUMAP2(component[i], factors, pf2Points, projs, ax[(2*i)+1])

    return f

