import numpy as np
import anndata
from parafac2 import parafac2_nd


def tensorFy(annD: anndata.AnnData, obsName: str) -> list[np.ndarray]:
    observation_vec = annD.obs_vector(obsName)
    sgUnique, sgIndex = np.unique(observation_vec, return_inverse=True)

    data_list = [annD[sgIndex == sgi, :].X for sgi in range(len(sgUnique))]
    return data_list


def pf2(X: anndata.AnnData, condition_name: str, rank: int, random_state=1):
    X_pf = tensorFy(X, condition_name)

    weight, factors, projs, _ = parafac2_nd(
        X_pf,
        rank=rank,
        random_state=random_state,
    )

    X.uns["Pf2_weights"] = weight
    X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"] = factors
    X.obsm["projections"] = np.concatenate(projs, axis=0)
    X.obsm["weighted_projections"] = X.obsm["projections"] @ X.uns["Pf2_B"]

    return X
