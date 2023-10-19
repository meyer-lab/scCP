"""
S2: Examining cell state for Pf2 on lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: visualize the component compostition by cell type
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotUMAP import plotCmpPerCellType, plotCmpUMAP, points


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 18), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of data
    X = openPf2(40, "Lupus")

    comps = [13, 14, 16, 26, 29, 32]
    for i, comp in enumerate(comps):
        # TODO: Fix plotCmpPerCellType
        # plotCmpPerCellType(weightedProjDF, comp, ax[(2*i)], outliers=False)
        plotCmpUMAP(X, comp, ax[(2 * i) + 1])

    points(X.obsm["embedding"], labels=X.obs["cell_type_broad"], ax=ax[14])
    ax[14].set(ylabel="UMAP2", xlabel="UMAP1")

    return f
