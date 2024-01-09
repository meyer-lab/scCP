"""
Thomson: XX
"""
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotUMAP import plotLabelsUMAP, plotCmpUMAP, plotGeneUMAP
from .commonFuncs.plotGeneral import (
    plotGenePerCellType,
    plotGenePerCategStatus,
    plotGeneFactors,
    gene_plot_cells,
    plot_cell_gene_corr,
    heatmapGeneFactors,
)
import scanpy as sc

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 18), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("factor_cache/Lupus.h5ad", backed="r")

    genes = ["PLBD1"]
    X = X.obs["Cell Type"] == ("B" or "NK")
    # X = sc.pp.subsample(X, fraction=0.01, random_state=0, copy=True)
    for i, gene in enumerate(genes):
        plotGenePerCategStatus(["SLE"], "lupus", gene, X, ax[i], obs = "SLE_status", mean=True, cellType="louvain")


    return f
