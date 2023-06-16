"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
import seaborn as sns
import pandas as pd
import math
from .common import subplotLabel, getSetup, plotDistDrug, plotDistGene, plotDistAllDrug
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from ..metric import distDrugDF, distGeneDF, distAllDrugDF


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 6), (3, 4))
    subplotLabel(ax)  # Add subplot labels

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()

    ranks = [1]
    
    Pf2s = [parafac2_nd(data, rank=rank, random_state=1, verbose=True) for rank in ranks]
    PCs = [PCA(n_components=rank).fit_transform(data.unfold()) for rank in ranks]
    
    drugs = ["Triamcinolone Acetonide"]
    drugDistanceDF = distDrugDF(data, ranks, Pf2s, PCs, drugs)
    plotDistDrug(drugDistanceDF, drugs, ax[0:1])
    
    genes = ["NKG7"]
    
    geneDistanceDF = distGeneDF(data, ranks, Pf2s, PCs, genes)
    plotDistGene(geneDistanceDF, genes, ax[1:2])


    allDrugDF = distAllDrugDF(data, Pf2s[0], PCs[0])
    plotDistAllDrug(allDrugDF, ranks[0], ax[2:3])
    
  
 
    # dim_red_var_drug(data, ranks, Pf2s, ["Budesonide", "Loteprednol etabonate", "Betamethasone Valerate", "Triamcinolone Acetonide"], ax[0:4])
    # dim_red_var_cell(data, ranks, Pf2s, ["NKG7", "CD79A", "CD3D", "LAD1"], 0.1, ax[4:8])

    # rank = 24
    # Pf2 = parafac2_nd(data, rank=rank, random_state=1, verbose=True)
    # pc = PCA(n_components=rank)
    # PC = pc.fit_transform(data.unfold())
    

    # all_drug_dist(data, Pf2, PC, ax[8])
    # all_marker_dist(data, Pf2, PC, 0.03, ax[9])

    return f
