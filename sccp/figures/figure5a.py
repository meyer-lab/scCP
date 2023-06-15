"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
import seaborn as sns
import pandas as pd
import math
from .common import subplotLabel, getSetup, flattenData
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 6), (1, 2))
    subplotLabel(ax)  # Add subplot labels

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()

    # ranks = [5, 10]

    # Pf2s = [parafac2_nd(data, rank=rank, random_state=1, verbose=True) for rank in ranks]

    # dim_red_var_drug(data, ranks, Pf2s, ["Dexrazoxane HCl (ICRF-187, ADR-529)"], ax[0:1])
    # dim_red_var_cell(data, ranks, Pf2s, ["NKG7", "GNLY"], 0.1, ax[1:3])

    rank = 25
    Pf2 = parafac2_nd(data, rank=rank, random_state=1, verbose=True)
    
    # factors = Pf2[1]
    # projs = Pf2[2]
    _, projDF, _ = flattenData(data, Pf2[1], Pf2[2])

    
    
    pc = PCA(n_components=rank)
    PC = pc.fit_transform(data.unfold())
    

# Rapamycin (Sirolimus)
# Mianserin HCl
# Masitinib (AB1010)
    yupDF = all_drug_dist(data, Pf2, PC, ax[0])
    
    yupDF["Cell Count"] = np.repeat(projDF.groupby("Drug").count()["Cmp. 1"].values, 2)
    
    print(np.repeat(projDF.groupby("Drug").count()["Cmp. 1"].values, 2))
    
    print(yupDF)
    
    sns.scatterplot(data=yupDF.loc[yupDF["Method"] == "PARAFAC2"], x="Cell Count", y="% Total Variance", ax=ax[0])
    sns.scatterplot(data=yupDF.loc[yupDF["Method"] == "PCA"], x="Cell Count", y="% Total Variance", ax=ax[1])
    
    # all_marker_dist(data, Pf2, PC, 0.03, ax[4])

    return f


def dim_red_var_drug(data, ranks, Pf2s, drugs, ax):
    """Plots normalized variance for either a variable or for a group of cells"""
    var_DF = pd.DataFrame()
    for ii, rank in enumerate(ranks):
        _, factors, projs, _ = Pf2s[ii]
        _, projDF, _ = flattenData(data, factors, projs)
        for drug in drugs:
            Pf2_all = projDF.values[:, 0:-1]
            Pf2_drug = projDF.loc[projDF.Drug == drug].values[:, 0:-1]

            pc = PCA(n_components=rank)
            PC_all = pc.fit_transform(data.unfold())
            PC_drug = PC_all[projDF.Drug == drug]

            Pf2_var = np.sum(np.std(Pf2_drug.astype(float), axis=0)) / np.sum(np.std(Pf2_all.astype(float), axis=0))
            PC_var = np.sum(np.std(PC_drug, axis=0)) / np.sum(np.std(PC_all, axis=0))
            var_DF = pd.concat([var_DF, pd.DataFrame({"Rank": [rank], "% Total Variance": Pf2_var, "Method": "PARAFAC2", "Drug": drug})])
            var_DF = pd.concat([var_DF, pd.DataFrame({"Rank": [rank], "% Total Variance": PC_var, "Method": "PCA", "Drug": drug})])
    
    # var_DF = var_DF.reset_index(drop=True)
    # for i, drug in enumerate(drugs):
    #     plot_DF = var_DF.loc[var_DF.Drug == drug]
    #     sns.lineplot(data=plot_DF, x="Rank", y="% Total Variance", hue="Method", ax=ax[i])
    #     ax[i].set(title=drug)
        
    return var_DF


def dim_red_var_cell(data, ranks, Pf2s, markers, cutoff, ax):
    """Plots normalized variance for either a variable or for a group of cells"""
    var_DF = pd.DataFrame()
    dist_DF = pd.DataFrame()
    for ii, rank in enumerate(ranks):
        _, factors, projs, _ = Pf2s[ii]
        dataDF, projDF, _ = flattenData(data, factors, projs)
        for marker in markers:
            dataDF[marker + " status"] = "Marker Negative"
            dataDF.loc[dataDF[marker] > cutoff, marker + " status"] = "Marker Positive"
            Pf2_all = projDF.values[:, 0:-1]
            Pf2_cell = projDF.loc[
                dataDF[marker + " status"] == "Marker Positive"
            ].values[:, 0:-1]

            pc = PCA(n_components=rank)
            PC_all = pc.fit_transform(data.unfold())
            PC_cell = PC_all[dataDF[marker + " status"] == "Marker Positive"]

            Pf2_pwise = centroid_dist(Pf2_cell) / centroid_dist(Pf2_all)
            PC_pwise = centroid_dist(PC_cell) / centroid_dist(PC_all)
            dist_DF = pd.concat([dist_DF, pd.DataFrame({"Rank": [rank], "Norm Centroid Distance": Pf2_pwise, "Method": "PARAFAC2", "Marker": marker})])
            dist_DF = pd.concat([dist_DF, pd.DataFrame({"Rank": [rank], "Norm Centroid Distance": PC_pwise, "Method": "PCA", "Marker": marker})])

    dist_DF = dist_DF.reset_index(drop=True)
    for i, marker in enumerate(markers):
        plot_DF = dist_DF.loc[dist_DF.Marker == marker]
        sns.lineplot(
            data=plot_DF, x="Rank", y="Norm Centroid Distance", hue="Method", ax=ax[i]
        )
        ax[i].set(title=marker)
        
        


def PCA_look(data, ranks, marker, cutoff, ax):
    """Plots normalized variance for either a variable or for a group of cells"""
    var_DF = pd.DataFrame()
    for rank in ranks:
        _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
        verbose=True,
        )
        dataDF, projDF, _ = flattenData(data, factors, projs)
        dataDF["Cell"] = "Other"
        dataDF.loc[dataDF[marker] > cutoff, "Cell"] = "Marker Positive"
        projDF["Cell"] = dataDF.Cell.values
        Pf2_all = projDF.values[:, 0:-2]
        Pf2_cell = projDF.loc[dataDF.Cell == "Marker Positive"].values[:, 0:-2]

    for i in np.arange(0, rank-1):
        sns.scatterplot(data=projDF, x="Cmp. " + str(i+1), y="Cmp. " + str(i+2), hue="Cell", ax=ax[i])


def inter_clust_dist(clustPoints, otherPoints):
    """Calculates mean distance between clust points and all others"""
    dists = []
    clustVec = np.tile(clustPoints, (otherPoints.shape[0], 1))
    otherPointsVec = np.repeat(otherPoints, np.repeat(clustPoints.shape[0], otherPoints.shape[0], 0), axis=0)
    dists = np.zeros(clustVec.shape[0])
    for i in range(clustVec.shape[0]):
        dists[i] = math.dist(clustVec[i, :], otherPointsVec[i, :])
    return np.mean(dists)


def centroid_dist(points):
    """Calculates mean distance between clust points and all others"""
    centroid = np.mean(points, axis=0)
    dist_vecs = np.linalg.norm(points - np.reshape(centroid, (1,-1)))
    
    # print(dist_vecs)
    
    # row_norms = np.linalg.norm(np.array(dist_vecs, dtype=np.float64), axis=0)
    # print(np.shape(row_norms))
    # a
    # lol = np.mean(np.abs(dist_vecs), axis=0)
    
    
    # print(row_norms)
    return dist_vecs


def all_drug_dist(data, Pf2, PC, ax):
    """Calculates mean distance between clust points and all others"""
    var_DF = pd.DataFrame()

    factors = Pf2[1]
    projs = Pf2[2]
    _, projDF, _ = flattenData(data, factors, projs)
    Pf2_all = projDF.values[:, 0:-1]


    for drug in projDF.Drug.unique():
        Pf2_drug = projDF.loc[projDF.Drug == drug].values[:, 0:-1]
        PC_drug = PC[projDF.Drug == drug]
        # print(drug)
        # print(Pf2_drug)
        # print(np.std(Pf2_drug.astype(float), axis=0))
        # print(np.sqrt(np.var(Pf2_drug, axis=0).astype(float)))
        # print(np.sum(np.sqrt(np.var(Pf2_drug, axis=0))))
        Pf2_var = np.sum(np.std(Pf2_drug.astype(float), axis=0)) / np.sum(np.std(Pf2_all.astype(float), axis=0))
        PC_var = np.sum(np.std(PC_drug, axis=0)) / np.sum(np.std(PC, axis=0))
        
        # print(drug)
        # print(Pf2_var)
        # print(PC_var)
        
        
        var_DF = pd.concat([var_DF, pd.DataFrame({"Drug": [drug], "% Total Variance": Pf2_var, "Method": "PARAFAC2",})])
        var_DF = pd.concat([var_DF, pd.DataFrame({"Drug": [drug], "% Total Variance": PC_var, "Method": "PCA"})])
    
    var_DF = var_DF.reset_index(drop=True)
    # sns.swarmplot(data=var_DF, x="Method", y="% Total Variance", size=2, ax=ax)
    # ax.set(title="All Drugs: Rank = " + str(PC.shape[1]))
    
    return var_DF


def all_marker_dist(data, Pf2, PC, cutoff, ax):
    """Calculates mean distance between clust points and all others"""
    dist_DF = pd.DataFrame()

    factors = Pf2[1]
    projs = Pf2[2]
    dataDF_init, projDF, _ = flattenData(data, factors, projs)
    Pf2_all = projDF.values[:, 0:-1]

    markers = [item for value in marker_genes.values() for item in (value if isinstance(value, list) else [value])]

    dataDF, projDF, _ = flattenData(data, factors, projs)
    Pf2_all = projDF.values[:, 0:-1]

    for marker in markers:
        if marker in dataDF_init.columns:
            dataDF[marker + " status"] = "Marker Negative"
            dataDF.loc[dataDF[marker] > cutoff, marker + " status"] = "Marker Positive"

    for marker in markers:
        if marker in dataDF_init.columns:
            Pf2_cell = projDF.loc[dataDF[marker + " status"] == "Marker Positive"].values[:, 0: -1]
            PC_cell = PC[dataDF[marker + " status"] == "Marker Positive"]
            

            Pf2_pwise = centroid_dist(Pf2_cell) / centroid_dist(Pf2_all)
            PC_pwise = centroid_dist(PC_cell) / centroid_dist(PC)
            dist_DF = pd.concat([dist_DF, pd.DataFrame({"Marker": [marker], "Norm Centroid Distance": Pf2_pwise, "Method": "PARAFAC2"})])
            dist_DF = pd.concat([dist_DF, pd.DataFrame({"Marker": [marker], "Norm Centroid Distance": PC_pwise, "Method": "PCA"})])
            
            print(marker)
            print(Pf2_pwise)
            print(PC_pwise)
        

    dist_DF = dist_DF.reset_index(drop=True)
    print(dist_DF)
    sns.swarmplot(data=dist_DF, x="Method", y="Norm Centroid Distance", size=2, ax=ax)
    ax.set(title="All Canonical Markers: Rank = " + str(PC.shape[1]))



marker_genes = {
    "Monocytes": ["CD14", "CD33", "LYZ", "LGALS3", "CSF1R", "ITGAX", "HLA-DRB1"],
    "Dendritic Cells": ["LAD1", "LAMP3", "TSPAN13", "CLIC2", "FLT3"],
    "B-cells": ["MS4A1", "CD19", "CD79A"],
    "T-helpers": ["TNF", "TNFRSF18", "IFNG", "IL2RA", "BATF"],
    "T cells": [
        "CD27",
        "CD69",
        "CD2",
        "CD3D",
        "CXCR3",
        "CCL5",
        "IL7R",
        "CXCL8",
        "GZMK",
    ],
    "Natural Killers": ["NKG7", "GNLY", "PRF1", "FCGR3A", "NCAM1", "TYROBP"],
    "CD8": ["CD8A", "CD8B"],
}
