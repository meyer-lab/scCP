"""
S2: Examining cell state for Pf2 on lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: visualize the cell state compostition by cell type/UMAP

import numpy as np
from .common import (
    subplotLabel,
    getSetup,
    plotCmpUMAP,
    plotCellStateViolins,
    plotUMAP_ct,
)
from ..imports.scRNA import load_lupus_data
from parafac2 import parafac2_nd
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.decomposition import PCA

def plotCompViolins(projections, cell_types, component: int, ax):
    all_cell_projs = pd.DataFrame(projections)
    comp_n = pd.concat([all_cell_projs.iloc[:, (component - 1)], cell_types], axis = 1)
    comp_n.columns.values[0] = "contribution"

    sns.violinplot(data = comp_n,
                   x = "cg_cov",
                   y = 'contribution',
                   hue = 'cg_cov',
                   dodge = False,
                   ax = ax)
    
    ax.set_title('Cell Type Contrib to Component ' + str(component))
    ax.tick_params(axis="x", rotation=90)
    ax.get_legend().remove()


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of data
    data, cell_types, _ = load_lupus_data()  # don't need to get patient color mappings
    rank = 40
    cellState = 17
    cmp = 17

    # run pf2
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
    )

    projs = np.concatenate(projs, axis=0)

    proj_B = projs @ factors[1]
    print(proj_B)
    print(proj_B.shape)

    # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1).fit(projs)

    # PCA dimension reduction
    pc = PCA(n_components=rank)
    pcaPoints = pc.fit_transform(data.unfold())
    pcaPoints = umap.UMAP(random_state=1).fit(pcaPoints)

    plotUMAP_ct(cell_types, pf2Points, ax[0])
    plotCmpUMAP(cellState, cmp, factors, pf2Points, projs, ax[1])
    plotCompViolins(proj_B, cell_types, cmp, ax[2])

    return f
