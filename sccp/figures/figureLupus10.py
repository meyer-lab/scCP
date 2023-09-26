"""
Determining differces in raw gene expression for lupus status
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: visualize the component compostition by cell type
from .common import subplotLabel, getSetup, flattenData
from ..imports.scRNA import load_lupus_data
from .commonFuncs.plotGeneral import plotGenePerCategCond, plotGenePerCategStatus
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((22, 25), (4, 5))

    # Add subplot labels
    subplotLabel(ax)

    # Import of data
    data, annData = load_lupus_data()
    
    
     
    condLabels = annData[["sample_ID", "SLE_status"]].drop_duplicates("sample_ID")
    condLabels = np.asarray(condLabels["SLE_status"])
    # print(condLabels)
    
    
    dataDF = flattenData(data)
    
    
    cellCount = dataDF.groupby(["Condition"]).size().values
    condNames=[]
    
   
    
    print(cellCount)
    print(len(cellCount))
    
    for i in range(len(data.X_list)):
        print(i)
        condNames = np.append(
            condNames, np.repeat(condLabels[i], cellCount[i])
        )
    
    dataDF["Condition2"] = condNames
    print(dataDF)
    cell_types = annData[["cell_type_broad", "SLE_status"]].reset_index(drop=True)
    dataDF["Cell Type"] = cell_types["cell_type_broad"].values

    # geneSet1 = ["HIST1H2AC", "PPBP"]
    # geneSet1 = ["HIST1H2AC", "PPBP", "CLU", "PF4", "TUBB1", "NRGN", "FCER1A", "CLEC10A", "HLA-DPA1", "HLA-DPB1"]
    # geneSet2 = ["IFI6", "MT2A", "LY6E", "ISG15", "TNFAIP3", "DUSP6", "ATP2B1", "KLF6", "SGK1", "EGR1"]
    
    
    # geneSet1 = ["ITGB1", "TNFRSF4", "CD69", "IL32", "CD40LG", "CRIP1", "B2M", "ANXA1", "GATA3", "LTB"]
    # geneSet2 = ["CD8B", "TIGIT", "ISG15", "GZMH", "RGS1", "IFI44L", "IFI6", "LY6E" , "CD8A"]
        
        
    # geneSet1 = ["S100A4", "B2M", "PTGER2", "CD74", "KLF6", "ANXA1", "LGALS1", "ZFP36", "CCL5", "HLA-DRB1"]
    # geneSet2 = ["FHIT", "RPL21", "CCR7", "TSHZ2", "PIK3IP1", "ADTRP", "PRKCQ-AS1", "CD40LG", "ITM2A"]
    
    
    geneSet1 = ["ISG15", "IFI6", "LY6E", "IFI44L", "IFIT3", "IFITM3", "IFIT1", "APOBEC3A", "IFIT2"]
    geneSet2 = ["RPL21", "S100A12", "IRS2", "VCAN", "MGST1", "AHNAK", "NRGN", "CSF3R", "IRAK3"]
    a = ["SLE"]
    
    
    
    plotGenePerCategCond(a, "Lupus", geneSet1, dataDF, ax[0:10])
    plotGenePerCategCond(a, "Lupus", geneSet2, dataDF, ax[10:20])

    return f