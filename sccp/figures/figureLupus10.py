"""
S2: Examining cell state for Pf2 on lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: visualize the component compostition by cell type
from .common import subplotLabel, getSetup, flattenWeightedProjs, openPf2, openUMAP, flattenData
from .commonFuncs.plotUMAP import plotCmpPerCellType, plotCmpUMAP, plotCellTypeUMAP
from ..imports.scRNA import load_lupus_data
import numpy as np


from .commonFuncs.plotGeneral import plotGenePerCellType, plotGenePerCategCond


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((25, 25), (5, 5))

    # Add subplot labels
    subplotLabel(ax)

    # Import of data
    data, annData = load_lupus_data()
    
    condLabels = annData[["sample_ID", "SLE_status"]].drop_duplicates("sample_ID")
    data.condition_labels = np.asarray(condLabels["SLE_status"])
    
    dataDF = flattenData(data)
    cell_types = annData[["cell_type_broad", "SLE_status"]].reset_index(drop=True)
    dataDF["Cell Type"] = cell_types["cell_type_broad"].values
    # dataDF = dataDF.loc[dataDF["Cell Type"] == "B"]
    geneSet1 = ["HIST1H2AC", "PPBP", "CLU", "PF4", "TUBB1", "NRGN", "FCER1A", "CLEC10A", "HLA-DPA1", "HLA-DPB1", "HLA-DRB1", "HLA-DRA"]
    geneSet2 = ["IFI6", "MT2A", "LY6E", "ISG15", "TNFAIP3", "DUSP6", "ATP2B1", "KLF6", "SGK1", "EGR1"]
    # geneSet1 = ["HIST1H2AC"]
    plotGenePerCategCond(["SLE"], "Lupus", geneSet1, dataDF, ax[0:13])
    plotGenePerCategCond(["SLE"], "Lupus", geneSet2, dataDF, ax[13:24])


    # # get cell types
    # cell_types = annData[["cell_type_broad", "SLE_status"]].reset_index(drop=True)
    # (
    #     _,
    #     factors,
    #     projs,
    # ) = openPf2(rank=rank, dataName="lupus", optProjs=True)
    
    # pf2Points = openUMAP(rank, "lupus", opt=True)

    # weightedProjDF = flattenWeightedProjs(lupus_tensor, factors, projs)
    # weightedProjDF["Cell Type"] = cell_types["cell_type_broad"].values
    # dataDF["Cell Type"] = cell_types["cell_type_broad"].values

    # comps = [13, 14, 16, 26, 29, 32]
    # for i, comp in enumerate(comps):
    #     plotCmpPerCellType(weightedProjDF, comp, ax[(2*i)], outliers=False)
    #     plotCmpUMAP(comp, factors, pf2Points, projs, ax[(2*i)+1])




   
    # dataDF = flattenData(lupus_tensor)

    # # get cell types
    # cell_types = obs[["cell_type_broad", "SLE_status"]].reset_index(drop=True)
    # (
    #     _,
    #     factors,
    #     projs,
    # ) = openPf2(rank=rank, dataName="lupus", optProjs=True)
    
    # pf2Points = openUMAP(rank, "lupus", opt=True)

    # weightedProjDF = flattenWeightedProjs(lupus_tensor, factors, projs)
    # weightedProjDF["Cell Type"] = cell_types["cell_type_broad"].values
    # dataDF["Cell Type"] = cell_types["cell_type_broad"].values
    
    return f