"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import subplotLabel, getSetup, plotCmpUMAP, openPf2
from ..imports.scRNA import ThompsonXA_SCGenes
from parafac2 import parafac2_nd
import umap 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 10), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    rank = 30

    weight, factors, projs = openPf2(rank, "Thomson")
    
    # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1).fit_transform(projs)
    
    cmp=30
    weightedProjs = projs @ factors[1]
    weightedProjs = weightedProjs[:, cmp-1]
       
    cellSkip = 15
    umap1 = pf2Points[::cellSkip, 0]
    umap2 = pf2Points[::cellSkip, 1]
    
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    weightedProjs = weightedProjs / np.max(np.abs(weightedProjs))
    psm = plt.pcolormesh([[-1, 1],[-1, 1]], cmap=cmap)
    
    weightedProjs =  weightedProjs[::cellSkip]

    ax[0].scatter(
            umap1,
            umap2,
            c=weightedProjs,
            cmap=cmap,
            s=0.2,
    )
    ax[0].set_xticks(np.arange(-5, 10, .5))
    ax[0].set_yticks(np.arange(-3, 12.5, .5))
    plt.colorbar(psm, ax=ax[0])
    
    # dentridit cells , X>5 
    
    # b cells greater than B 9.5
    
    # nk cells, x less then -.75
    # btween 5 and 9 fory 

    #macrophagse -5 and 5 for x
    # betwen -2.5 and 4.5 for y 

    
    # cellState = np.arange(25, 31, 1) 
    # component = np.copy(cellState)
    
    # cmp = 30
    # weightedProjs = projs @ factors[1]
    # weightedProjs = weightedProjs[:, cmp-1]
    
    # idx = np.argwhere(weightedProjs<0)
    
    # print(idx)
    # print(len(weightedProjs[idx]))


    
    
    
    
    
    # idx = np.argwhere(weightedProjs>0)
    # print(idx)
    # print(len(weightedProjs[idx]))
    
    
    # for i in range(len(cellState)):
    #     plotCmpUMAP(cellState[i], component[i], factors, pf2Points, projs, ax[i])

    return f


