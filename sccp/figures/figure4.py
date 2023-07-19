"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (subplotLabel, 
getSetup, plotFactors, 
plotR2X, plotCV,
plotCondFactorsReorder, plotWeight)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import numpy as np
import pandas as pd

import os
from os.path import join

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 25), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    rank = 1

    weight, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
    )
    
    savePf2(weight, factors, projs, "Thomson_scRNAseq")
    
    # factorss = [factors[0], factors[1], factors[2]]
    
    # proj = []
    # for i in range(len(projs)):
    #     proj = np.append(proj, projs[i])
        
    


    # plotFactors(factors, data, ax[0:3], reorder=(0, 2), trim=(2,), saveGenes=False)
    # plotCondFactorsReorder(factors, data, ax[3])
    # plotWeight(weight, ax[4])

    # plotCV(data, rank+3, trainPerc=0.75, ax=ax[2])
    # plotR2X(data, rank+3, ax=ax[3])

    return f

def savePf2(weight, factors, projs, dataName):
    rank = len(weight)
    np.save(join(path_here, "data/WeightCmp"+str(rank)+"_"+dataName+".npy"), weight)
    for i in range(3):
        np.save(join(path_here, "data/Factors"+str(i)+"_Cmp"+str(rank)+"_"+dataName+".npy"), factors[i])
        
    np.save(join(path_here, "data/ProjCmp"+str(rank)+"_"+dataName+".npy"), np.concatenate(projs, axis=0))