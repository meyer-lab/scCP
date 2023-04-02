from os.path import dirname
import numpy as np
import pandas as pd
import xarray as xa
from scipy.stats import linregress
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn import preprocessing
import anndata
import scanpy as sc


path_here = dirname(dirname(__file__))


def xarrayIfy(annD, obsV, limit_cells=None):
    sgUnique, sgIndex, sgCounts = np.unique(
        obsV, return_inverse=True, return_counts=True
    )
    if limit_cells is None:
        limit_cells = np.max(sgCounts)

    X = xa.DataArray(
        data=np.zeros((len(sgUnique), limit_cells, annD.shape[1]), dtype=np.float32),
        dims=["sgRNA", "Cells", "Genes"],
        coords=[
            sgUnique,
            np.arange(limit_cells),
            annD.var_names,
        ],
    )

    for sgi in range(len(sgUnique)):
        x_temp = annD[sgIndex == sgi, :]
        assert x_temp.shape[0] == sgCounts[sgi]

        if x_temp.shape[0] > limit_cells:
            x_temp = x_temp[:limit_cells, :]

        X[sgi, 0 : x_temp.shape[0], :] = x_temp.X.toarray()

    return X


def import_perturb_RPE(limit_cells: int=100):
    ds_disk = anndata.read_h5ad("/opt/andrew/rpe1_normalized_singlecell_01.h5ad")

    sgRNAs = ds_disk.obs_vector("sgID_AB")
    X = xarrayIfy(ds_disk, sgRNAs, limit_cells)

    # These genes have nans for some reason
    X = xa.concat([X[:, :, 0:773], X[:, :, 774:]], dim="Genes")
    X = xa.concat([X[:, :, 0:7002], X[:, :, 7003:]], dim="Genes")

    # Do not allow problematic values
    assert np.all(np.isfinite(X))
    return X


def import_thompson_drug():
    """Imports cell readings from scRNA-seq data from PBMCs from PopAlign paper."""

    # Cell barcodes, sample id of treatment and sample number (33482, 3)
    metafile = pd.read_csv("sccp/data/meta.csv")

    # Cell barcodes (33482)
    barcodes = pd.read_csv("sccp/data/barcodes.tsv", sep="\t", header=None, names=("cell_barcode", ))

    # Left merging should put the barcodes in order
    metafile = pd.merge(barcodes, metafile, on="cell_barcode", how='left', validate="one_to_one")

    # h5ad is simplified version of mtx format
    # data = sc.read_10x_mtx("/opt/andrew/Thompson", var_names='gene_symbols', make_unique=True)
    # data.write('thompson.h5ad', compression="gzip")
    data = anndata.read_h5ad("/opt/andrew/thompson.h5ad")

    data.obs["Drugs"] = pd.Categorical(metafile["sample_id"])
    return data


def mu_sigma_normalize(X: anndata.AnnData, scalingfactor: float):
    """Calculates the mu and sigma for every gene and returns
    means, sigmas, and dataframe filtered for genes expressed
    in > 0.1% of cells."""
    assert np.all(np.isfinite(X.X.data))
    X.X = X.X.todense()

    X = X[:, np.mean(X.X > 0, axis=0) > 0.001]
    X.X /= np.sum(X.X, axis=0)

    # Only operating on the data works because 0 ends up as 0 here
    X.X = np.log10((scalingfactor * X.X) + 1)
    means = np.mean(X.X, axis=0)
    cv = np.std(X.X, axis=0) / means

    return X, np.log10(means + 1e-10), np.log10(cv + 1e-10)


def gene_import(offset_value=1.0):
    """Imports gene data from PopAlign and performs gene filtering process."""
    genesDF = import_thompson_drug()
    X, logmean, logstd = mu_sigma_normalize(genesDF, scalingfactor=1000)

    if offset_value != 1.0:
        slope, intercept, _, _, _ = linregress(logmean, logstd)

        above_idx = logstd > logmean * slope + intercept + np.log10(offset_value)
        X = X[:, above_idx]

    return X


def ThompsonXA_SCGenes(offset=1.0):
    """Turns filtered and normalized cells into an Xarray."""
    anndta = gene_import(offset_value=offset)
    # anndta = assign_celltype(anndta)

    # Assign cells a count per-experiment so we can reindex
    X = xarrayIfy(anndta, anndta.obs_vector("Drugs"))
    X.name = "data"

    return xa.merge([X], compat="no_conflicts")


def assign_celltype(adata):
    """Assigning cell types via scanpy and SVM."""
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=0.75)
    sc.tl.rank_genes_groups(adata, groupby="leiden")
    sc.tl.umap(adata)
    print(adata.obs)
    #adata.obs = adata.obs.replace(clust_names)
    #adata.obs.columns = ["Cell Type"]
    #adata = drug_SVM(adata, genes_list)
    assert False
    return adata


def drug_SVM(save_data, genes):
    """Retrieves cell types from perturbed data"""
    completeDF = pd.DataFrame(data=save_data.X)
    completeDF.columns = genes
    completeDF["Cell Type"] = save_data.obs.values
    drugDF = completeDF.loc[
        (completeDF["Cell Type"] == "T Helpers") | (completeDF["Cell Type"] == "None")
    ]
    trainingDF = pd.DataFrame()
    for key in training_markers:
        cell_weight = (
            completeDF.loc[completeDF["Cell Type"] == key].shape[0]
            / completeDF.shape[0]
        )
        concatDF = drugDF.sort_values(by=training_markers[key]).tail(
            n=int(np.ceil(50 * cell_weight) + 5)
        )
        concatDF["Cell Type"] = key
        trainingDF = pd.concat([trainingDF, concatDF])

    le = preprocessing.LabelEncoder()
    le.fit(trainingDF["Cell Type"].values)
    svm = make_pipeline(preprocessing.StandardScaler(), SVC(gamma="auto"))
    svm.fit(
        trainingDF.drop("Cell Type", axis=1),
        le.transform(trainingDF["Cell Type"].values),
    )
    drugDF = drugDF.drop("Cell Type", axis=1)
    drugDF["Cell Type"] = le.inverse_transform(svm.predict(drugDF.values))
    save_data.obs.loc[
        (save_data.obs["Cell Type"] == "T Helpers")
        | (save_data.obs["Cell Type"] == "None"),
        "Cell Type",
    ] = drugDF["Cell Type"].values
    return save_data


clust_names = {
    "0": "NK",
    "1": "Monocytes",
    "2": "Monocytes",
    "3": "Monocytes",
    "4": "None",
    "5": "T Cells",
    "6": "CD8 T Cells",
    "7": "NK",
    "8": "T Cells",
    "9": "T Helpers",
    "10": "Dendritic Cells",
    "11": "B Cells",
    "12": "Monocytes",
    "13": "Monocytes",
    "14": "Monocytes",
}


clust_list = [
    "NK",
    "Monocytes ",
    "Monocytes  ",
    "Monocytes   ",
    "None",
    "T Cells",
    "CD8 T Cells",
    "NK ",
    "T Cells ",
    "T helpers",
    "Dendritic Cells",
    "B Cells",
    " Monocytes",
    "  Monocytes",
    "   Monocytes",
]


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

training_markers = {
    "Monocytes": ["CD14"],
    "Dendritic Cells": ["LAD1"],
    "B Cells": ["MS4A1"],
    "T Cells": ["CD3D"],
    "NK": ["NKG7"],
    "CD8 T Cells": ["CD8A"],
}
