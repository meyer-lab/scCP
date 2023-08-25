"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (subplotLabel, getSetup)
import numpy as np
import pandas as pd
import scipy.io
import anndata


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (4, 4))

    # Add subplot labels
    subplotLabel(ax)
    
    # barcodes = pd.read_csv("/opt/andrew/HamadCITEseq/control/barcodes.tsv.gz", sep="\t", header=None, names=("cell_barcode",))
    #     features = pd.read_csv("/opt/andrew/HamadCITEseq/"+files[i]+"/features.tsv.gz", sep="\t", header=None)
    #     data = pd.DataFrame(scipy.io.mmread("/opt/andrew/HamadCITEseq/"+files[i]+"/matrix.mtx.gz").todense())
    
    #     data["Expression Type"] = features.iloc[:, 2].values
    #     data["Expression Name"] = features.iloc[:, 1].values
        
    #     geneDF = data.loc[data["Expression Type"] == "Gene Expression"].drop(columns="Expression Type").reset_index(drop=True)
    #     proteinDF = data.loc[data["Expression Type"] == "Antibody Capture"].drop(columns="Expression Type").reset_index(drop=True)
    #     geneNames = geneDF["Expression Name"].values
    
    files = ["ic_pod1", "ic_pod1", "ic_pod1", "ic_pod1"]
    
 
 
    for i in range(len(files))
    # barcodes = pd.read_csv("/opt/andrew/HamadCITEseq/control/barcodes.tsv.gz", sep="\t", header=None, names=("cell_barcode",))
        features = pd.read_csv("/opt/andrew/HamadCITEseq/"+files[i]+"/features.tsv.gz", sep="\t", header=None)
        data = pd.DataFrame(scipy.io.mmread("/opt/andrew/HamadCITEseq/"+files[i]+"/matrix.mtx.gz").todense())
    
        data["Expression Type"] = features.iloc[:, 2].values
        data["Expression Name"] = features.iloc[:, 1].values
        
        geneDF = data.loc[data["Expression Type"] == "Gene Expression"].drop(columns="Expression Type").reset_index(drop=True)
        proteinDF = data.loc[data["Expression Type"] == "Antibody Capture"].drop(columns="Expression Type").reset_index(drop=True)
        geneNames = geneDF["Expression Name"].values
        
        
    
    
    geneDF = np.transpose(geneDF.drop(columns="Expression Name"))
    condNames = []
    np.repeat("Control", geneDF.shape[0])
    print(geneDF.to_numpy())
    # geneDF.columns = geneNames
    
    # print(geneDF)
    
    
    # features = pd.read_csv("/opt/andrew/HamadCITEseq/ic_pod1/features.tsv.gz", sep="\t", header=None)
    # data = pd.DataFrame(scipy.io.mmread("/opt/andrew/HamadCITEseq/ic_pod1/matrix.mtx.gz").todense())
    
    # data["Expression Type"] = features.iloc[:, 2].values
    # data["Expression Name"] = features.iloc[:, 1].values
    
    # geneDF = data.loc[data["Expression Type"] == "Gene Expression"].drop(columns="Expression Type").reset_index(drop=True)
    # proteinDF = data.loc[data["Expression Type"] == "Antibody Capture"].drop(columns="Expression Type").reset_index(drop=True)
    
    # print(geneDF)
    
 
    # geneNames = geneDF["Expression Name"].values
    geneDF = np.transpose(geneDF.drop(columns="Expression Name"))
    # geneDF.columns = geneNames
    
    
    
 
    
    # np.vstack((a,b))


    # assert gnees/proteins are the same for all experiments 

    
    
    # adata = anndata.AnnData(X = geneDF.to_numpy())
    # adata.var_names = geneDF.columns
    
    # adata
    
    # print(len(adata.var_names))
    # adata.var = "Gene"
  
    
    # data = np.transpose(data)
    # data.columns = features.iloc[:, 1].values
    
    
    # # data["Expression Type"] = features.iloc[:, 2].values
    # # data["Expression Name"] = features.iloc[:, 1].values
    # # data.reset_index().set_index("Expression Name", inplace=True)
    # data["Condition"] = np.repeat("Control", len(barcodes.values))
    # data
    
    # data = np.transpose(data)
    
    # print(data)
    
    # data.columns = barcodes.values
    # data = 
    # data.columns = features.iloc[:,1].values
    # ad.AnnData(X=data.to_m())


    # adata = anndata.AnnData(X = np.transpose(data).to_numpy(), var="Genes, var_names = 
    # print(adata)
    # print(adata.X)
    # # var = features.iloc[:,1].values)
    # adata.obs["Expression Type"] = features.iloc[:, 1].values
    

    
    
    # print(adata.X)
    # data["Expression Type"] = features.iloc[:, 2].values
    # data["Expression Name"] = features.iloc[:, 1].values
    # data = np.transpose(data)
    # print(data)



 

    
    
    return f