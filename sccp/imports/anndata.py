"""
Fucntions to indicate if format used from Theis Lab
"""
"""
Tests based on Theis lab to check for AnnData format and output is correct for Pf2 
"""
import anndata
from scib import utils
from ..parafac2 import parafac2_nd
from tensorly.parafac2_tensor import parafac2_to_slices


def pf2Theis(adata, batch, hvg=None, rank=20):
    """Run Pf2 for adata"""
    utils.check_sanity(adata, batch, hvg)
    split = utils.split_batches(adata.copy(), batch)

    XX = [x.X for x in split]
    weights, factors, projs, _ = parafac2_nd(XX, rank=rank, random_state=1, verbose=True, tol=1e-9)

    reconst = parafac2_to_slices((weights, factors, projs))

    for ii, recon in enumerate(reconst):
        split[ii].X = recon
        split[ii].obsm['X_emb'] = projs[ii]

    return anndata.concat(split)
