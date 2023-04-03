from os.path import dirname
import numpy as np
import pandas as pd
import xarray as xa
from scipy.stats import linregress
import anndata


path_here = dirname(dirname(__file__))


def xarrayIfy(annD, obsV):
    sgUnique, sgIndex, sgCounts = np.unique(
        obsV, return_inverse=True, return_counts=True
    )

    data = list()

    for sgi in range(len(sgUnique)):
        x_temp = annD[sgIndex == sgi, :]
        assert x_temp.shape[0] == sgCounts[sgi]

        data.append(xa.DataArray(
            name=sgUnique[sgi],
            data=x_temp.X.toarray(),
            dims=[f"cells_{sgUnique[sgi]}", "genes"],
            coords=[
                np.arange(x_temp.shape[0]),
                annD.var_names,
            ],
        ))

    return xa.merge(data)


def import_perturb_RPE():
    ds_disk = anndata.read_h5ad("/opt/andrew/rpe1_normalized_singlecell_01.h5ad")

    sgRNAs = ds_disk.obs_vector("sgID_AB")
    X = xarrayIfy(ds_disk, sgRNAs)

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

    # Assign cells a count per-experiment so we can reindex
    data = xarrayIfy(anndta, anndta.obs_vector("Drugs"))
    return data
