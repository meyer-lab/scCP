""""""

canonicalGenes = {
    "CD4 Memory T ": ["CD3D", "CREM"],
    "CD4 Native T ": ["CD3D", "SELL", "GIMAP5"],
    "Activated T ": ["CREM", "CACYBP"],
    "Natural Killer": ["GNLY", "NKG7"],
    "CD8 T": ["CD3D", "NKG7", "CD8A"],
    "B": ["CD79A"],
    "Activated B": ["MIR155HG", "NEM1"],
    "CD16 Monocytes": ["FCGR3A", "VMO1"],
    "CD14 Monocytes": ["CCL2", "S100A9"],
    "Dentritic Cells": ["HLA-DQA1", "GPR183"],
    "Megakaryocyte": ["HBA2", "HBB"],
    "Plasmacytoid Dendritic Cells": ["TSPAN13,", "IGJ"]}



def distAllGeneDF(data, Pf2s, PCs):
    """Plots normalized variance for either a variable or for a group of cells"""
    distDF = pd.DataFrame([])
    
    factors = Pf2s[1]
    projs = Pf2s[2]
    dataDF, projDF, _ = flattenData(data, factors, projs)
    pf2All = projDF.values[:, 0:-1]
    pcaAll = PCs
    
    markers = [item for value in marker_genes.values() for item in (value if isinstance(value, list) else [value])]

    datDFcopy = dataDF.copy()
    for marker in markers:
        if marker in datDFcopy.columns:
            dataDF.loc[dataDF[marker] > 0.03, marker + " status"] = "Marker Positive"
            
    for marker in markers:
        if marker in datDFcopy.columns:
            pf2Gene = projDF.loc[dataDF[marker + " status"] == "Marker Positive"].values[:, 0: -1]
            pcaGene = pcaAll[dataDF[marker + " status"] == "Marker Positive"]

            pf2Dist = centroid_dist(pf2Gene) / centroid_dist(pf2All)
            pcaDist = centroid_dist(pcaGene) / centroid_dist(pcaAll)