"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
    openUMAP,
    flattenData,
    flattenWeightedProjs,
)
from .commonFuncs.plotGeneral import plotGenePerCellType, plotGenePerCategCond
from .commonFuncs.plotUMAP import (
    plotCellTypeUMAP,
    plotCmpPerCellType,
    plotCmpUMAP,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..imports.gating import gateThomsonCells
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy import stats



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    dataDF = flattenData(data)

    rank = 30
    dataDF["Cell Type"] = gateThomsonCells(rank=rank, saveCellTypes=False)

    _, factors, projs = openPf2(rank, "Thomson")
    # pf2Points = openUMAP(rank, "Thomson", opt=False)

    # plotCellTypeUMAP(pf2Points, dataDF, ax[0])

    weightedProjDF = flattenWeightedProjs(data, factors, projs)
    weightedProjDF["Cell Type"] = dataDF["Cell Type"].values
    weightedProjDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    dataDF.sort_values(by=["Condition", "Cell Type"], inplace=True)
    
    threshold = -.3
    cmp = 30
    Wass_KL_Dist(threshold, cmp, data, dataDF, weightedProjDF)

    comps = [5, 12, 20, 30]
    
    
    # -.3 for b cells that are more greater


    return f




def Wass_KL_Dist(threshold, cmp, data, dataDF, weightedProjDF):
    """Finds markers which have average greatest difference from other cells"""
    for gene in data.variable_labels:
        geneAvg = np.mean(dataDF[gene])   
        cmpWeights = weightedProjDF[["Cell Type", "Cmp. "+str(cmp)]]
        print(cmpWeights)
        if threshold > 0: 
            idxCells = np.argwhere(cmpWeights["Cmp. "+str(cmp)] > threshold)
            idxOpp = np.argwhere(cmpWeights["Cmp. "+str(cmp)] < threshold)
        else:
            idxCells = np.argwhere(cmpWeights["Cmp. "+str(cmp)] < threshold)
            idxOpp = np.argwhere(cmpWeights["Cmp. "+str(cmp)] > threshold)
        
        geneDF = dataDF[["Cell Type", "Condition", gene]]
        
        
        print(idxCells)
        print(np.shape(idxCells))
        
        targCells = geneDF.iloc[idxCells.flatten(), :] 
        targCells[gene] /= geneAvg
        offCells = geneDF.iloc[idxOpp.flatten(), :] 
        offCells[gene] /= geneAvg
        
        targCells = targCells[gene].values
        offCells = offCells[gene].values
        
        # geneDFf[geneDF.index.difference(idxCells), ["Cell Type", "Condition", gene]] / geneAvg
        
        print(targCells)
        print(offCells)
        
        # geneDF[(geneDF < weight).any(1)] / geneAvg
        
        # if np.mean(targCells) > np.mean(offCells):
        
        print(np.shape(targCells))
        print(np.shape(offCells))
        
        offCells = offCells[0:5]
            
        kdeTarg = KernelDensity(kernel='gaussian').fit(np.reshape(targCells, (-1, 1)))
        kdeOffTarg = KernelDensity(kernel='gaussian').fit(np.reshape(offCells, (-1, 1)))
        minVal = np.minimum(np.min(targCells), np.min(offCells)) - 10
        maxVal = np.maximum(np.max(targCells), np.max(offCells)) + 10
        outcomes = np.arange(minVal, maxVal + 1).reshape(-1, 1)
        distTarg = np.exp(kdeTarg.score_samples(outcomes))
        distOffTarg = np.exp(kdeOffTarg.score_samples(outcomes))
        KL_div = stats.entropy(distOffTarg.flatten() + 1e-200, distTarg.flatten() + 1e-200, base=2)
        markerDF = pd.concat([markerDF, pd.DataFrame({"Marker": [gene], "Wasserstein Distance": stats.wasserstein_distance(targCellMark, offTargCellMark), "KL Divergence": KL_div})])
        print(markerDF)
        
        
    #     cmpWeights 
        
    #     specificcels = only choose cells whose weighted avergea is alge [gnees] /markAvg
    #     off target = all other cells and divide by the markavg
    #     if mean is greater than mean of off targe mean
            

    # markerDF = pd.DataFrame(columns=["Marker", "Cell Type", "Amount"])
    # for marker in CITE_DF.loc[:, ((CITE_DF.columns != 'CellType1') & (CITE_DF.columns != 'CellType2') & (CITE_DF.columns != 'CellType3') & (CITE_DF.columns != 'Cell'))].columns:
    #     markAvg = np.mean(CITE_DF[marker].values)
    #     if markAvg > 0.0001:
    #         targCellMark = CITE_DF.loc[CITE_DF["CellType2"] == targCell][marker].values / markAvg
    #         offTargCellMark = CITE_DF.loc[CITE_DF["CellType2"] != targCell][marker].values / markAvg
    #         if np.mean(targCellMark) > np.mean(offTargCellMark):
    #             kdeTarg = KernelDensity(kernel='gaussian').fit(targCellMark.reshape(-1, 1))
    #             kdeOffTarg = KernelDensity(kernel='gaussian').fit(offTargCellMark.reshape(-1, 1))
    #             minVal = np.minimum(targCellMark.min(), offTargCellMark.min()) - 10
    #             maxVal = np.maximum(targCellMark.max(), offTargCellMark.max()) + 10
    #             outcomes = np.arange(minVal, maxVal + 1).reshape(-1, 1)
    #             distTarg = np.exp(kdeTarg.score_samples(outcomes))
    #             distOffTarg = np.exp(kdeOffTarg.score_samples(outcomes))
    #             KL_div = stats.entropy(distOffTarg.flatten() + 1e-200, distTarg.flatten() + 1e-200, base=2)
    #             markerDF = pd.concat([markerDF, pd.DataFrame({"Marker": [marker], "Wasserstein Distance": stats.wasserstein_distance(targCellMark, offTargCellMark), "KL Divergence": KL_div})])

    # corrsDF = pd.DataFrame()
    # for i, distance in enumerate(["Wasserstein Distance", "KL Divergence"]):
    #     ratioDF = markerDF.sort_values(by=distance)
    #     posCorrs = ratioDF.tail(numFactors).Marker.values
    #     corrsDF = pd.concat([corrsDF, pd.DataFrame({"Distance": distance, "Marker": posCorrs})])
    #     markerDF = markerDF.loc[markerDF["Marker"].isin(posCorrs)]
    #     sns.barplot(data=ratioDF.tail(numFactors), y="Marker", x=distance, ax=ax[i], color='k')
    #     ax[i].set(xscale="log")
    #     #ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45)
    # if RNA:
    #     ax[0].set(title="Wasserstein Distance - RNA")
    #     ax[1].set(title="KL Divergence - RNA")
    # else:
    #     ax[0].set(title="Wasserstein Distance - Surface Markers")
    #     ax[1].set(title="KL Divergence - Surface Markers")
    # return corrsDF
