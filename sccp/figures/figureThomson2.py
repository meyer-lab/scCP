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
    plotGenePerCategCond,
    plotGeneFactors,
    gene_plot_cells,
)
from ..gating import gateThomsonCells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((24, 24), (4, 4))
    # ax, f = getSetup((16, 16), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("factor_cache/Thomson.h5ad", backed="r")

    # plotfms(X, 30, ax[0])
    # plotR2X_pf2(X, 15, ax[1])

    gateThomsonCells(X)

    plotLabelsUMAP(X, "Cell Type", ax[0])
    plotLabelsUMAP(X, "Cell Type2", ax[1])

    # plotLabelsUMAP(X, "doublet", ax[2], cmap='viridis')
    # plotLabelsUMAP(X, "doublet_score", ax[3], cmap='viridis')

    plotCmpUMAP(X, 12, ax[2], 0.2)  # pDCs
    plotCmpUMAP(X, 19, ax[3], 0.2)  # Alpro
    plotCmpUMAP(X, 20, ax[4], 0.2)  # Gluco

    plotGeneFactors(12, X, ax[6], geneAmount=10, top=True)
    plotGeneFactors(12, X, ax[7], geneAmount=10, top=False)
    plotGeneFactors(19, X, ax[8], geneAmount=10, top=True)
    plotGeneFactors(19, X, ax[9], geneAmount=10, top=False)
    plotGeneFactors(20, X, ax[10], geneAmount=10, top=True)
    plotGeneFactors(20, X, ax[11], geneAmount=10, top=False)
    # plotGeneFactors(26, X, ax[9], geneAmount=10, top=False)
    # plotGeneFactors(26, X, ax[10], geneAmount=10, top=True)

    geneSet1 = ["NKG7", "GNLY", "GZMB", "GZMH", "PRF1", "CD3D"]
    geneSet2 = ["MS4A1", "CD79A", "CD79B", "TNFRSF13B", "BANK1"]

    genes = [geneSet1, geneSet2]
    for i in range(len(genes)):
        plotGenePerCellType(genes[i], X, ax[i + 11])

    glucs = [
        "Betamethasone Valerate",
        "Loteprednol etabonate",
        "Budesonide",
        "Triamcinolone Acetonide",
        "Meprednisone",
    ]
    geneSet3 = ["CD163"]
    #geneSet3 = ["NDRG2"]
    plotGenePerCategCond(glucs, "Gluco", geneSet3, X, [ax[13]])

    geneSet3 = ["C1QC"]
    #geneSet3 = ["NDRG2"]
    #plotGenePerCategCond(glucs, "Gluco", geneSet3, X, [ax[14]])

    X.obs["Drug Type"] = "Other"
    X.obs.loc[X.obs.Condition.isin(glucs), "Drug Type"] = "Gluco"

    #X_genes = X[:, ["NDRG2", "CD163"]].to_memory()
    #gene_plot_cells(X_genes, hue="Condition", unique=glucs, ax=ax[14], kde=False, average=True)

    #X_genes = X[:, ["NDRG2", "CD163"]].to_memory()
    #gene_plot_cells(X_genes, hue="Cell Type", ax=ax[14], kde=False, average=True)

    X_genes = X[:, ["CCL7", "CD163"]].to_memory()
    gene_plot_cells(X_genes, hue="Cell Type", ax=ax[14], kde=False, average=True)

    return f
