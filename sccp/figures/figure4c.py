"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (subplotLabel, getSetup, 
                     plotCmpUMAP, openPf2, 
                     flattenData, flattenWeightedProjs, 
                     plotCellTypeUMAP, plotCmpPerCellType, 
                     plotGenePerCellType)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..imports.gating import gateThomsonCells
import umap 
import os
from os.path import join
import pandas as pd


path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 16), (4, 5))

    # Add subplot labels
    subplotLabel(ax)
    
    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    dataDF = flattenData(data)
    
    rank = 30
    dataDF["Cell Type"] = gateThomsonCells(rank=rank, saveCellTypes=False)

    # _, factors, projs = openPf2(rank, "Thomson")
    # pf2Points = umap.UMAP(random_state=1).fit(projs)
    
    # # plotCellTypeUMAP(pf2Points, dataDF, ax[0])
        
    # weightedProjDF = flattenWeightedProjs(data, factors, projs)
    # weightedProjDF["Cell Type"] = dataDF["Cell Type"].values
    # weightedProjDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    # dataDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    
    # comps = ["Cmp. 5", "Cmp. 10", "Cmp. 12", "Cmp. 20", "Cmp. 30"]
    # for i, comp in enumerate(comps):
    #     plotCmpPerCellType(weightedProjDF, comp, ax[i+1])

    # genes = ["NKG7", "GNLY", "MS4A1", "CD79A", "GPR183", "HLA-DQA1"]
    # genes = ["NKG7", "GNLY","KLRD1", "GZMA", "GZMH", "MS4A1", "CD79A","CD79B"]
    genes = ["CD163", "ADORA3","MS4A6A", "RNASE1", "MTMR11", "NDRG2", "CRABP2", "AK8", "GPC4","CCL22"]
    
    # genes = ["VPREB3", "CD79A","FAM111B", "HOPX", "SLC30A3", "CCNB1", "GADD45A", "SLC40A1", "CDC20","ITIH3"]
    
    for i, gene in enumerate(genes):
        # plotGenePerCellType(dataDF[(dataDF["Condition"].isin(["Budesonide", "CTRL1", "Alprostadil", "Cyclosporine", "Tazarotene", "Dexrazoxane HCl (ICRF-187, ADR-529)"]))], gene, ax[i])
        
        plotGenePerCellType(dataDF[(dataDF["Condition"].isin(["Loteprednol etabonate", "Betamethasone Valerate", "Triamcinolone Acetonide", "Budesonide", "Meprednisone", "CTRL1", "CTRL2", "CTRL3"]))], gene, ax[i])
        # plotGenePerCellType(dataDF[(dataDF["Condition"].isin(["Dexrazoxane HCl (ICRF-187, ADR-529)"]))], gene, ax[i])
    
    
    return f