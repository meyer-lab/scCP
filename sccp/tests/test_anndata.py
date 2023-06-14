"""
Test that we have correctly setup a Theis lab method function.
"""
import anndata


def test_anndata():
    ds_disk = anndata.read_h5ad("/opt/andrew/anndata_test.h5ad")

    # example call
    # function(ds_disk, ds_disk.obs["batch"])

    # TODO: Add test for return values
