"""
Hamad CITEseq dataset
"""
from .common import (
    subplotLabel,
    getSetup,
    plotFactors,
    plotWeight,
    plotDrugUMAP,
    flattenData
)
from ..imports.citeseq import import_citeseq, combine_all_citeseq
from parafac2 import parafac2_nd
import umap
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 10), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    # combine_all_citeseq(saveAdata=True)

    data = import_citeseq()
    rank = 20
    
    
    weight, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
    )

    # plotFactors(factors, data, ax[0:3], reorder=(0, 2), trim=(2,), saveGenes=False)
    # plotWeight(weight, ax[3])
    
    
       # Find cells associated with drugs
    pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))
    dataDF = flattenData(data)
    drugs = [
        "control",
        "sc_pod1",
        "sc_pod7",
        "ic_pod1",
        "ic_pod7"
    ]
    plotDrugUMAP(drugs, "Pf2", dataDF["Condition"].values, pf2Points, ax[0:6])
    # plotDrugUMAP(drugs, "PCA", dataDF["Condition"].values, pcaPoints, ax[6:8])
    

    return f
