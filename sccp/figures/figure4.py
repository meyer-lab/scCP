"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (subplotLabel, 
getSetup, plotFactorsRelaxed, 
plotR2X, plotCV, plotWeight,
openPf2, savePf2)
from ..imports.scRNA import ThompsonXA_SCGenesAD, tensorFy
from ..imports.gating import gateThomsonCells
from parafac2 import parafac2_nd

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 16), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenesAD()
    rank = 30
    data.obs["cell_type"] = gateThomsonCells(rank)

    for type in data.obs.cell_type.unique()[4:]:
        print(type)
        print(len(data))
        data_subset = data[data.obs.cell_type != type]
        print(len(data_subset))
        data_subset = tensorFy(data_subset, "Drugs")
        weight, factors, projs, _ = parafac2_nd(
        data_subset,
        rank=rank,
        random_state=1,
        )
        plotFactorsRelaxed(factors, data_subset, ax[0:3], reorder=(0, 2), trim=(2,), saveGenes=True)
        plotWeight(weight, ax[3])
        return f

    # weight, factors, projs, _ = parafac2_nd(
    #     data,
    #     rank=rank,
    #     random_state=1,
    # )
    
    # savePf2(weight, factors, projs, "Thomson")
    # weight, factors, _ = openPf2(rank, "Thomson")

    

    # plotFactorsRelaxed(factors, data, ax[0:3], reorder=(0, 2), trim=(2,), saveGenes=True)
    # plotWeight(weight, ax[3])

    # plotCV(data, rank+3, trainPerc=0.75, ax=ax[2])
    # plotR2X(data, rank+3, ax=ax[3])

    return f

