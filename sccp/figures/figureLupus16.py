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
    plot2GenePerCategStatus,
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

    genes = ["CDKN1A", "APOBEC3A", "LPAR6", "DPYSL2", "ABI3", "PLBD1", "RBP7", "S100A12", "CDA", "S100A8"] #43
    # genes = ["SGK1", "MAFB", "DUSP6", "C5AR1", "RGCC", "CLEC4E", "IRS2", "LRRK2", "IER3", "PDE4B"] #22
    # genes = ["RGS1", "PTGER4", "PMAIP1", "NR4A2", "CD83", "CLDND1", "AC092580.4", "BANK1", "PTGER2", "ESF1"] #30
    # genes = ["IL8", "MGST1", "IRS2", "NR4A2", "S100A12", "CDA", "APOBEC3A", "ISG15", "IFITM3", "MX1"] #39
    # genes = ["IFI27", "IFITM3", "IFI6", "ISG15", "APOBEC3A", "RETN", "CD8A", "PTGER4", "ESF1", "ANXA2R"] # 48
    X = X.to_memory()
    # ind = X.obs["Cell Type"] == ("cM" or "ncM")
    # X = X[ind, :]
    # ind
    
    
    
    cmp = 39
    ind = X.obsm["weighted_projections"] > .1
    X = X[ind[:, cmp-1], :]
   
   
    genes = ["IL8", "APOBEC3A"] 
    plot2GenePerCategStatus(["SLE"], "lupus", genes[0],genes[1], X, ax[0], obs = "SLE_status", mean=True, cellType="Cell Type")


    # ind = X.obsm["Cell"] < -.1
    # X = X[ind[:, cmp-1], :]
   
    # # X = sc.pp.subsample(X, fraction=0.01, random_state=0, copy=True)
    # for i, gene in enumerate(genes):
    #     plotGenePerCategStatus(["SLE"], "lupus", gene, X, ax[i], obs = "SLE_status", mean=True, cellType="Cell Type")



    return f

