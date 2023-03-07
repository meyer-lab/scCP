"""
# This creates Figure 3, typing Thompson drug data.
# """
import pandas as pd
from .common import subplotLabel, getSetup
# from gmm.scImport import gene_import
# import scanpy as sc
import numpy as np
from copy import copy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import preprocessing


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 15), (3, 3))
    subplotLabel(ax)

    # Add subplot labels
    # geneDF = gene_import(offset=1, filter=False)
    # drugCol = geneDF.Drug
    # geneDF.drop(columns=["Drug"], axis=1, inplace=True)
    # genes_list = geneDF.columns
    # adata = sc.AnnData(geneDF)
    # sc.pp.pca(adata, svd_solver='arpack')
    # sc.pp.neighbors(adata)
    # sc.tl.leiden(adata, resolution=0.75)
    # sc.tl.rank_genes_groups(adata, groupby='leiden')
    # marker_matches = sc.tl.marker_gene_overlap(adata, marker_genes)
    # print(marker_matches)
    # sc.tl.umap(adata)
    # sc.pl.umap(adata, ax=ax[0])
    # savedata = copy(adata)
    # savedata.obs = savedata.obs.replace(clust_names)
    # savedata.obs.columns = ["Cell Type"]
    # savedata = drug_SVM(savedata, genes_list)
    # savedata.obs.to_csv("gmm/data/ThompsonCellTypes.csv")
    # adata.rename_categories('leiden', clust_list)
    # adata.obs["Drug"] = drugCol.values
    # adata.obs["Cell Type"] = savedata.obs
    # sc.pp.subsample(adata, fraction=0.1, random_state=0)
    # sc.pl.umap(adata, color='leiden', legend_loc='on data', title='', frameon=False, ax=ax[1])
    # sc.pl.umap(adata, color='Cell Type', title='', legend_fontsize=3, ax=ax[2])
    # sc.pl.umap(adata, color=['CD14'], ax=ax[3])
    # sc.pl.umap(adata, color=['LAD1'], ax=ax[4])
    # sc.pl.umap(adata, color=['MS4A1'], ax=ax[5])
    # sc.pl.umap(adata, color=['CD3D'], ax=ax[6])
    # sc.pl.umap(adata, color=['IL2RA'], ax=ax[7])
    # sc.pl.umap(adata, color=['NKG7'], ax=ax[8])


    return f