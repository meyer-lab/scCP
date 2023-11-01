"""
Lupus: Plot 2 Pf2 factors for conditions
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotGeneral import gene_plot_cells, gene_plot_conditions, geneSig_plot_cells, wproj_plot_cells
import numpy as np
import pandas as pd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 24), (6, 2))

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    X = openPf2(rank=rank, dataName="Lupus")
    print(X)
    wproj_plot_cells(X, [13, 26], ax[8], average=None)
    wproj_plot_cells(X, [13, 26], ax[9], average=True)
    wproj_plot_cells(X, [16, 32], ax[10], average=None)
    wproj_plot_cells(X, [16, 32], ax[11], average=True)
    gene_plot_cells(X, ["PPBP", "FHIT"], hue="SLE_status", ax=ax[0], kde=True)
    gene_plot_cells(X, ["PPBP", "FHIT"], hue="Cell Type", ax=ax[1])
    gene_plot_cells(X, ["PPBP", "FHIT"], hue="Cell Type", ax=ax[2], average=True)
    gene_plot_cells(X, ["PPBP", "FHIT"], hue="Cell Type", ax=ax[3], average=True, unique="Progen")
    #gene_plot_conditions(X, "patient", ["PPBP", "FHIT"], ax[4], hue="SLE_status")
    geneSig_plot_cells(X, [13, 26], hue="SLE_status", ax=ax[4], kde=True)
    geneSig_plot_cells(X, [13, 26], hue="Cell Type", ax=ax[5])
    geneSig_plot_cells(X, [13, 26], hue="SLE_status", ax=ax[6], average=True)
    geneSig_plot_cells(X, [13, 26], hue="Cell Type", ax=ax[7], average=True)

    
    return f
