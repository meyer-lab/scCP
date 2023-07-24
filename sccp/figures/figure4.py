"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (subplotLabel, 
getSetup, plotFactors, 
plotR2X, plotCV, plotWeight,
openPf2, savePf2)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import numpy as np

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 12), (2, 2))
    # 10/4

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    print(data.condition_labels)
    rank = 30

    # weight, factors, projs, _ = parafac2_nd(
    #     data,
    #     rank=rank,
    #     random_state=1,
    # )
    
    # savePf2(weight, factors, projs, "Thomson")
    weight, factors, projs = openPf2(rank, "Thomson")
    
    listDrugs = []
    listDrugs = np.concatenate((np.repeat("Other", 3), ["Prostaglandin"], ["Glucorticoid"], ["Other"],  
                                ["Glucorticoid"], ["Other"], np.repeat(["Control"], 6), np.repeat("Other", 2),
                                ["Calcineurin Inhbitor"], np.repeat("Other", 3), ["mTOR Inhibitor"], np.repeat("Other", 7), 
                                ["Glucorticoid"], ["Other"], ["Glucorticoid"], np.repeat("Other", 7), ["mTOR Inhibitor"],  
                                np.repeat("Other", 4), ["Glucorticoid"], np.repeat("Other", 2)))
    
    
    
    listColors = ["k", "blueviolet", "darkorange", 
                "g",  "aqua","deeppink"]
    
    
    # print(listDrugs)
    # listDrugs[listDrugs == "Other"] = "k"
    # print(listDrugs)
    
    print(np.unique(listDrugs))
    
    for i, drug in enumerate(np.unique(listDrugs)):
        listDrugs[listDrugs == drug] = listColors[i]
        
    print(listDrugs)

    plotFactors(factors, data, ax[0:3], reorder=(0, 2), trim=(2,), row_colors=listDrugs)
    # plotWeight(weight, ax[3])

    # plotCV(data, rank+3, trainPerc=0.75, ax=ax[2])
    # plotR2X(data, rank+3, ax=ax[3])

    return f

