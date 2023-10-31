"""
Thomson: Creative ways to visualize single cells 
"""
import numpy as np
from .common import getSetup, openPf2
from .commonFuncs.plotUMAP import plotCmpPerCellType, plotLabelsUMAP, plotGenesFactorsUMAP, plotLabelsAveUMAP, plotPerGeneWP
from ..gating import gateThomsonCells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((14, 16), (4, 3))

    rank = 40
    X = openPf2(rank, "Lupus")


    cmp1 = 13
    cmp2 = 26

    plotLabelsUMAP(X, labelType="Cell Type",  ax=ax[0], cmp1=cmp1, cmp2=cmp2, type="WP")
    plotLabelsUMAP(X, labelType="SLE_status",  ax=ax[1], cmp1=cmp1, cmp2=cmp2, type="WP")
    ax[0].set(xlabel=f"Weighted Proj Cmp. {cmp1}", ylabel=f"Weighted Proj Cmp. {cmp2}")
    ax[1].set(xlabel=f"Weighted Proj Cmp. {cmp1}", ylabel=f"Weighted Proj Cmp. {cmp2}")
    
    plotLabelsUMAP(X, labelType="Cell Type",  ax=ax[2], cmp1=cmp1, cmp2=cmp2, type="WGene")
    plotLabelsUMAP(X, labelType="SLE_status",  ax=ax[3], cmp1=cmp1, cmp2=cmp2, type="WGene")
    ax[2].set(xlabel=f"Weighted Gene Cmp. {cmp1}", ylabel=f"Weighted Gene Cmp. {cmp2}")
    ax[3].set(xlabel=f"Weighted Gene Cmp. {cmp1}", ylabel=f"Weighted Gene Cmp. {cmp2}")
    
    
    plotLabelsAveUMAP(X, labelType="Cell Type",  ax=ax[4], cmp1=cmp1, cmp2=cmp2, stats="Mean", labelType2="SLE_status", type="WP")
    plotLabelsAveUMAP(X, labelType="Cell Type",  ax=ax[5], cmp1=cmp1, cmp2=cmp2, stats="Var", labelType2="SLE_status", type="WP")
    ax[4].set(xlabel=f"Weighted Proj Cmp. {cmp1}", ylabel=f"Weighted Proj Cmp. {cmp2}", title="Mean")
    ax[5].set(xlabel=f"Weighted Proj Cmp. {cmp1}", ylabel=f"Weighted Proj Cmp. {cmp2}", title="Var")
      
    plotLabelsAveUMAP(X, labelType="Cell Type",  ax=ax[6], cmp1=cmp1, cmp2=cmp2, stats="Mean", labelType2="SLE_status", type="WGene")
    plotLabelsAveUMAP(X, labelType="Cell Type",  ax=ax[7], cmp1=cmp1, cmp2=cmp2, stats="Var", labelType2="SLE_status", type="WGene")
    ax[6].set(xlabel=f"Weighted Gene Cmp. {cmp1}", ylabel=f"Weighted Gene Cmp. {cmp2}", title="Mean")
    ax[7].set(xlabel=f"Weighted Gene Cmp. {cmp1}", ylabel=f"Weighted Gene Cmp. {cmp2}", title="Var")

    plotPerGeneWP(X, labelType="Cell Type",  ax=ax[8], cmp1=cmp1, cmp2=cmp2, stats="Mean", labelType2="Cell Type", gene="PPBP")
    plotPerGeneWP(X, labelType="Cell Type",  ax=ax[9], cmp1=cmp1, cmp2=cmp2, stats="Mean", type="Ave", labelType2="SLE_status", gene="PPBP")
    plotPerGeneWP(X, labelType="SLE_status",  ax=ax[10], cmp1=cmp1, cmp2=cmp2, stats="Mean", labelType2="Cell Type", gene="PPBP")
    ax[8].set(xlabel=f"Weighted Proj Cmp. {cmp1}", ylabel="PPBP")
    ax[9].set(xlabel=f"Weighted Proj Cmp. {cmp1}", ylabel="PPBP", title="Mean")
    ax[10].set(xlabel=f"Weighted Proj Cmp. {cmp1}", ylabel="PPBP")

    plotGenesFactorsUMAP(X, ax=ax[11], cmp1=cmp1, cmp2=cmp2)
    ax[11].set(xlabel=f"Gene Factors Cmp. {cmp1}", ylabel=f"Gene Factors Cmp. {cmp2}")

    return f
    
    