"""
Hamad CITEseq dataset
"""
import numpy as np
import umap
from parafac2 import parafac2_nd

from .common import (
    subplotLabel,
    getSetup,
)
from ..imports.citeseq import import_citeseq, import_citeseqProt, combine_all_citeseqProt
from .commonFuncs.plotUMAP import (
    plotCmpUMAP, plotGeneUMAP
)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((50, 50), (5, 5))

    # Add subplot labels
    subplotLabel(ax)
    # data = combine_all_citeseqProt()
    # rank = 40

    data = import_citeseq()
    dataDF = import_citeseqProt()
    
    genes = ["MsRt.CD29"]
    rank=40
    
    
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
    )

    pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))
    
    plotGeneUMAP(genes, "Pf2", pf2Points, dataDF, ax[0:2])

    # _, factors, projs, _ = parafac2_nd(
    #     data,
    #     rank=rank,
    #     random_state=1,
    # )

    # # pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))

    # component = np.arange(1, 25, 1)

    # for i in range(len(component)):
    #     plotCmpUMAP(
    #         component[i], factors, pf2Points, np.concatenate(projs, axis=0), ax[i]
    #     )


    # plotGeneUMAP(genes, decomp, points, dataDF, axs):
    return f 
