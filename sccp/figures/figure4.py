"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
import numpy as np
from .common import (
    subplotLabel,
    getSetup,
    plotFactors,
    plotProj,
    plotR2X_CV
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
from ..crossVal import CrossVal
from ..decomposition import R2X


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 8), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    rank = 5

    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
    )


    # flattened_projs = np.concatenate(projs, axis=0)
    # idxx = np.random.choice(flattened_projs.shape[0], size=200, replace=False)

    # plotFactors(factors, data, ax[0:2], reorder=(0, 2), trim=(2,))
    
    plotR2X_CV(data, rank, trainPerc=0.75, ax=ax[3])
    
    
    
    

    # plotProj(flattened_projs[idxx, :], ax[3:5])
    


    # plotR2X(data, 30, ax[5])

    # plotCrossVal(data.X_list, 30, ax[6], trainPerc=0.75)

    return f
