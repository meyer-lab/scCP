"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
import seaborn as sns
import pandas as pd
from .common import (
    subplotLabel,
    getSetup,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import umap 



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes(offset=1.0)

    _, factors, projs, _ = parafac2_nd(
        data,
        rank=13,
    )

    dataDF, projDF = flattenData(data, factors, projs)

    # UMAP dimension reduction
    cmpNames = [f"Cmp. {i}" for i in np.arange(1, factors[0].shape[1] + 1)]
    umapReduc = umap.UMAP()
    umapPoints = umapReduc.fit_transform(projDF[cmpNames].to_numpy())
     
    # Mono1, Mono2, NK, DC, CD8, CD4, B
    genes = ["CD14", "FCGR3A", "NKG7", "CST3", "CD8B", "IL7R", "MS4A1"]
    plotGeneUMAP(genes, umapPoints, dataDF, ax[0:7])

    # Find cells associated with drugs
    drugs = ["CTRL2", "Triamcinolone Acetonide", "Budesonide", "Betamethasone Valerate", "Dexrazoxane HCl (ICRF-187, ADR-529)"]
    plotDrugUMAP(drugs, dataDF["Drug"].values, umapPoints, ax[7:12])

    return f

def flattenData(data, factors, projs):
    cellCount = []
    for i in range(factors[0].shape[0]):
        cellCount = np.append(cellCount, projs[i].shape[0])
    
    flatProjs = np.empty([int(np.sum(cellCount)), projs[0].shape[1]])
    flatData = np.empty([int(np.sum(cellCount)), len(data.variable_labels)])
    cellStart = [0]; drugNames = []
    
    for i in range(factors[0].shape[0]):
        cellStart = np.append(cellStart, cellStart[i] + cellCount[i])
        flatProjs[int(cellStart[i]): int(cellStart[i+1])] = projs[i]
        flatData[int(cellStart[i]): int(cellStart[i+1])] = data.X_list[i]
        drugNames = np.append(drugNames, np.repeat(data.condition_labels[i], cellCount[i]))
    
    cmpNames = [f"Cmp. {i}" for i in np.arange(1, factors[0].shape[1] + 1)]
    projDF = pd.DataFrame(data=flatProjs, columns = cmpNames)
    dataDF = pd.DataFrame(data=flatData, columns = data.variable_labels)
    projDF["Drug"] = drugNames
    dataDF["Drug"] = drugNames
    
    return dataDF, projDF

def plotGeneUMAP(genes, umapPoints, dataDF, axs):
    for i, genez in enumerate(genes):
        geneList = dataDF[genez].to_numpy()
        umapDF = pd.DataFrame({"UMAP1": umapPoints[::20, 0],
                "UMAP2": umapPoints[::20, 1],
                genez: geneList[::20],
            })
        sns.scatterplot(data=umapDF, x="UMAP1", y="UMAP2", hue=genez, s=5, ax=axs[i])
        axs[i].set(xlim=(-10, 20), ylim=(-15, 20))
        
    return 

def plotDrugUMAP(drugs, totaldrugs, umapPoints, axs):
    for i, drugz in enumerate(drugs):
        drugList = np.asarray(totaldrugs == drugz).astype(int)
        umapDF = pd.DataFrame({"UMAP1": umapPoints[::20, 0],
                "UMAP2": umapPoints[::20, 1],
                drugz: drugList[::20],
            })
        sns.scatterplot(data=umapDF, x="UMAP1", y="UMAP2", hue=drugz, s=5,  palette="muted", ax=axs[i])
        
    return
    
 