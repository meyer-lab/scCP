"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (subplotLabel, 
getSetup, plotFactorsRelaxed, 
plotR2X, plotCV, plotWeight,
openPf2, savePf2, plotCellTypeUMAP, flattenData, openUMAP)
from ..imports.scRNA import ThompsonXA_SCGenesAD, tensorFy
from ..imports.gating import gateThomsonCells
from parafac2 import parafac2_nd
import umap
import numpy as np
import json

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((36,32), (7, 7))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenesAD()
    rank = 30
    data.obs["cell_type"] = gateThomsonCells(rank)
    i = 0
    r2x_values = {}

    for cell_type in data.obs.cell_type.unique()[4:]:
        for drug in data.obs.Drugs.unique():
            print(cell_type)
            print(len(data))
            print(drug)
            data_subset = data[(data.obs.cell_type != cell_type) | (data.obs.Drugs != drug)]
            print(len(data_subset))
            dataDF = tensorFy(data_subset, "Drugs")
            weight, factors, projs, r2x = parafac2_nd(
                dataDF,
                rank=rank,
                random_state=1,
                )
            dataDF = flattenData(dataDF)
            dataDF['Cell Type'] = data_subset.obs['cell_type'].values
            # plotFactorsRelaxed(factors, data_subset, ax[0:3], reorder=(0, 2), trim=(2,), saveGenes=True)
            plotCellTypeUMAP(umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0)), dataDF, ax[i])
            print(f'R2X while dropping {cell_type} across {drug} condition: {r2x}')
            r2x_values[cell_type, drug] = r2x
            i += 1
        with open('./sccp/data/r2x_values.json', 'w+') as fp:
            fp.write(str(r2x_values))
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

