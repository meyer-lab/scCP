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
    plot2GenePerCategStatusCellType,
    plotGeneFactors,
    gene_plot_cells,
    plot_cell_gene_corr,
    heatmapGeneFactors,
)
import scanpy as sc
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.axes import Axes
from pacmap import PaCMAP
import scanpy as sc
from ..imports import import_lupus

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 18), (3, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted.h5ad", backed="r")

    # genes = ["APOBEC3A", "ISG15", "MX1", "IFI44", "MX2", "MGST1", "CAT", "LTA4H", "TGFBI", "IL8"] # 21 .06
    # genes = ["IFI27", "IFI6", "IFITM3", "ISG15", "IFI44L", "RETN", "CD8A", "RGCC", "PTGER4", "ANXA2R"] # 28 .06 
    # genes = ["IER3", "CLEC4E", "ALDH2", "PLBD1", "RGS18", "SGK1", "MAFB", "DUSP6", "RGCC", "CD83"] # 10 -.05
    # genes = ["IER3", "MGST1", "CLEC4E", "PLBD1", "LPAR6", "G0S2", "CDKN1A", "APOBEC3A", "IL1B", "CD83"] # 4 .06

    
     
    X = X.to_memory()
    # genes = [["CPPED1", "ALDH2"], ["CLEC4E", "RETN"], ["SGK1", "DUSP6"], ["MAFB", "IL8"]] # cmp 10
    # for i in range(len(genes)):
    #     plot2GenePerCategStatusCellType(["SLE"], "lupus", genes[i][0],genes[i][1], X, ax[i], obs = "SLE_status", celltype="cM", mean=True, cellType="Cell Type")
    #     ax[i].set_title("cM Average Expression Per Patient")

    # for i in range(len(genes)):
    #     plot2GenePerCategStatus(["SLE"], "lupus", genes[i][0],genes[i][1], X, ax[i+4], obs = "SLE_status", mean=True, cellType="Cell Type")
        
    genes = [["RHOB", "LRRK2"], ["ASGR1", "C9orf72"], ["IL1B", "IER3"], ["IL8", "MIR24-2"]] # cmp 24
    for i in range(len(genes)):
        plot2GenePerCategStatusCellType(["SLE"], "lupus", genes[i][0],genes[i][1], X, ax[i], obs = "SLE_status", celltype="ncM", mean=True, cellType="Cell Type")
        ax[i].set_title("cM Average Expression Per Patient")

    for i in range(len(genes)):
        plot2GenePerCategStatus(["SLE"], "lupus", genes[i][0],genes[i][1], X, ax[i+4], obs = "SLE_status", mean=True, cellType="Cell Type")    
    
    
    
    
    # genes = [["APOBEC3A", "ISG15"], ["MX1", "IFI44"], ["MGST1", "IL8"], ["LTA4H", "A1BG"]] # cmp 22 
    # for i in range(len(genes)):
    #     plot2GenePerCategStatusCellType(["SLE"], "lupus", genes[i][0],genes[i][1], X, ax[i], obs = "SLE_status", celltype="cM", mean=True, cellType="Cell Type")
    #     ax[i].set_title("cM Average Expression Per Patient")

    # for i in range(len(genes)):
    #     plot2GenePerCategStatus(["SLE"], "lupus", genes[i][0],genes[i][1], X, ax[i+4], obs = "SLE_status", mean=True, cellType="Cell Type")
    
    
    # genes = [["PLBD1", "RBP7"], ["CDA", "CLEC4E"], ["CDKN1A", "RETN"], ["APOBEC3A", "TNFAIP2"]] # cmp 8
    # for i in range(len(genes)):
    #     plot2GenePerCategStatusCellType(["SLE"], "lupus", genes[i][0],genes[i][1], X, ax[i], obs = "SLE_status", celltype="cM", mean=True, cellType="Cell Type")
    #     ax[i].set_title("cM Average Expression Per Patient")

    # for i in range(len(genes)):
    #     plot2GenePerCategStatus(["SLE"], "lupus", genes[i][0],genes[i][1], X, ax[i+4], obs = "SLE_status", mean=True, cellType="Cell Type")
    
    
    # genes = [["PLBD1", "RBP7"], ["CDA", "CLEC4E"], ["CDKN1A", "RETN"], ["APOBEC3A", "TNFAIP2"]] # cmp 20 
    # for i in range(len(genes)):
    #     plot2GenePerCategStatusCellType(["SLE"], "lupus", genes[i][0],genes[i][1], X, ax[i], obs = "SLE_status", celltype="cM", mean=True, cellType="Cell Type")
    #     ax[i].set_title("cM Average Expression Per Patient")

    # for i in range(len(genes)):
    #     plot2GenePerCategStatus(["SLE"], "lupus", genes[i][0],genes[i][1], X, ax[i+4], obs = "SLE_status", mean=True, cellType="Cell Type")
    
    # genes = ["FGFBP2", "DUSP2"]
    # plot2GenePerCategStatus(["SLE"], "lupus", genes[0],genes[1], X, ax[0], obs = "SLE_status", mean=True, cellType="Cell Type")
    
    # genes = ["CPPED1", "DUSP6"]
    # plot2GenePerCategStatus(["SLE"], "lupus", genes[0],genes[1], X, ax[1], obs = "SLE_status", mean=True, cellType="Cell Type")
    
    # genes = ["PLBD1", "CDKN1A"]
    # plot2GenePerCategStatus(["SLE"], "lupus", genes[0],genes[1], X, ax[2], obs = "SLE_status", mean=True, cellType="Cell Type")
    
    # genes = ["RBP7", "RETN"]
    # plot2GenePerCategStatus(["SLE"], "lupus", genes[0],genes[1], X, ax[3], obs = "SLE_status", mean=True, cellType="Cell Type")

    # genes = ["FGFBP2", "DUSP2", "CPPED1", "DUSP6", "RBP7", "RETN","PLBD1", "CDKN1A"]
    # for i, gene in enumerate(np.ravel(genes)):
    #     plotGenePerCategStatus(["SLE"], "lupus", gene, X, ax[i+4], obs = "SLE_status", mean=True, cellType="Cell Type")


    return f