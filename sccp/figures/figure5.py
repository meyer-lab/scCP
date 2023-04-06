"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
import seaborn as sns
import xarray as xa
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
    ax, f = getSetup((12, 12), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes(offset=1.0)

    _, factors, projs, _ = parafac2_nd(
        data,
        rank=3,
    )

    dataDF, projDF = flattenData(data, factors, projs)

    # UMAP dimension reduction
    cmpNames = [f"Cmp. {i}" for i in np.arange(1, factors[0].shape[1] + 1)]
    umapReduc = umap.UMAP()
    umapPoints = umapReduc.fit_transform(projDF[cmpNames].to_numpy())
     
    # DC, NK, CD8, Mono, CD4] Want b cells and t helprs    
    genes = ["LAD1", "NKG7", "CD8A", "CD33", "CCR7"]
    plotGeneUMAP(genes, umapPoints, dataDF, ax[0:5])

    # Find cells associated with drugs
    drugs = ["Triamcinolone Acetonide", "Budesonide", "Betamethasone Valerate", "Dexrazoxane HCl (ICRF-187, ADR-529)"]
    plotDrugUMAP(drugs, data.condition_labels, umapPoints, ax[5:9])
 
    
   
    
    # druglist[druglist == True] = 1
    # druglist[druglist == False] = 0
    # print(druglist)
    # print(np.shape(druglist))
    # for i, drugz in enumerate(drugs):
    #     flatData = flat_data.where(flat_data.Drug == drugz).to_numpy()
    #     flatData = np.nan_to_num(flatData[0, :], nan=-1)
    #     flatData[flatData != -1] = 1

    #     # Creating DF for figure
    #     umapDF = pd.DataFrame({"UMAP1": umapPoint s[:, 0],
    #             "UMAP2": umapPoints[:, 1],
    #             drugz: flatData,
    #         })
    #     sns.scatterplot(data=umapDF, x="UMAP1", y="UMAP2",hue=drugz, s=0.5, ax=ax[i])
        
    # print(np.shape(drugs))  
    # print(np.shape(totalProjs))
    
    # print(np.hstack((totalProjs,np.reshape(drugs, [-1, 1]))))
        
    
    
    
    # DF = pd.DataFrame([])
    # for i in range(factors[0].shape[0]): # for each drug
        
    #     df = pd.DateFrame([data=projs[i], columns=)
    #     DF = pd.concat([DF, df],axis=1)
        
    # data.condition_labels
    #     projs
    # #     totalProjs[i, :] = projs[0]
    # flatProjs = np.concatenate(projs, axis=0)

    # # idxx = np.random.choice(flatProjs.shape[0], size=200, replace=False)
    # # flatProjs = flatProjs[idxx, :]
    # proj = np.array(projs)
    # print(np.array(projs))
    # print(projs.to_numpy())
    # print(np.shape(proj))
    # projs = xa.DataArray(
    #     projs,
    #     dims=["Drug", "Cell", "Cmp"],
    #     coords=dict(
    #         Drug=data.coords["Drug"],
    #         Cell=data.coords["Cell"],
    #         Cmp=[f"Cmp. {i}" for i in np.arange(1, projs.shape[2] + 1)],
    #     ),
    
    #     name="projections",
    # )
    # projs = xa.merge([projs, data["Cell Type"]], compat="no_conflicts")

    # flattened_projs = projs.stack(AllCells=("Drug", "Cell"))
    # flat_data = data["data"].stack(AllCells=(("Drug", "Cell")))
  
    # # Remove empty slots
    # nonzero_index = np.any(flattened_projs["projections"].to_numpy() != 0, axis=0)
    # flattened_projs = flattened_projs.isel(AllCells=nonzero_index) 
    # flat_data = flat_data.isel(AllCells=nonzero_index) 
    
    # # UMAP dimension reduction
    # umapReduc = umap.UMAP()
    # umapPoints = umapReduc.fit_transform(flattened_projs["projections"].to_numpy().T)
    
    # # Find cells associated with drugs
    # drugs = ["Triamcinolone Acetonide", "Budesonide", "Betamethosone Valerate", "Dexrazoxane HCl (ICRF-187, ADR-529)"]
    
    # for i, drugz in enumerate(drugs):
    #     flatData = flat_data.where(flat_data.Drug == drugz).to_numpy()
    #     flatData = np.nan_to_num(flatData[0, :], nan=-1)
    #     flatData[flatData != -1] = 1

    #     # Creating DF for figure
    #     umapDF = pd.DataFrame({"UMAP1": umapPoints[:, 0],
    #             "UMAP2": umapPoints[:, 1],
    #             drugz: flatData,
    #         })
    # #     sns.scatterplot(data=umapDF, x="UMAP1", y="UMAP2",hue=drugz, s=0.5, ax=ax[i])
        
    
    # genes = ["LAD1", "NKG7", "CD8A", "CD33", "CCR7"]
    # # DC, NK, CD8, Mono, CD4]
    # for i, genez in enumerate(genes):
    #     gene_values = flat_data.sel(Gene=genez).to_numpy()
    #     tl = ax[i+4].scatter(umapPoints[::25, 0], umapPoints[::25, 1], c=gene_values[::25], cmap ="cool", s=0.5)
    #     f.colorbar(tl, ax=ax[i+4])
    #     ax[i+4].set_xlabel("UMAP1")
    #     ax[i+4].set_ylabel("UMAP2")
    #     ax[i+4].set_title(genez)

    
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
        axs[i].set(xlim=(-10, 20), ylim=(-15, 15))
        
    return 

def plotDrugUMAP(drugs, totaldrugs, umapPoints, axs):
    for i, drugz in enumerate(drugs):
        drugList = np.asarray(totaldrugs == drugz).astype(int)
        umapDF = pd.DataFrame({"UMAP1": umapPoints[::100, 0],
                "UMAP2": umapPoints[::100, 1],
                drugz: drugList[::100],
            })
        sns.scatterplot(data=umapDF, x="UMAP1", y="UMAP2", hue=drugz, s=5, ax=axs[i])
    
 