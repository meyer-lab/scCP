import os
from os.path import join
import numpy as np
from .parafac2 import _parafac2_rec_error, parafac2 as pf2
from tensorly.decomposition import parafac2


path_here = os.path.dirname(os.path.dirname(__file__))

def plotR2X(tensor, rank, datatype, ax, runPf2 = False):
    """Creates R2X plot for parafac2 tensor decomposition"""
    if runPf2 == True: 
        rank_vec = np.arange(1, rank + 1)
        pf2_error = np.empty(len(rank_vec))
        for i in range(len(rank_vec)):
            if len(tensor.shape) == 3:
                weights, factors, projs = parafac2(
                    tensor.to_numpy(),
                    rank=rank,
                    n_iter_max=10,
                    normalize_factors=True,
                    verbose=True)
            
            else: 
                weights, factors, projs = pf2(
                    tensor.to_numpy(),
                    rank=rank,
                    n_iter_max=10,
                    nn_modes=(0, 1, 2),
                    verbose=True)
            
            pf2_error[i] = _parafac2_rec_error(tensor, [weights, factors, projs])
                 
        if datatype == "IL2":
            np.save(join(path_here, "sccp/data/IL2_Pf2_Errors.npy"), pf2_error)  
        elif datatype == "CoH":
            np.save(join(path_here, "sccp/data/CoH_Pf2_Errors.npy"), pf2_error)  
        elif datatype == "scRNA":
            np.save(join(path_here, "sccp/data/scRNA_Pf2_Errors.npy"), pf2_error)  
    
    else: 
        if datatype == "IL2":
            pf2_error = np.load(join(path_here, "sccp/data/IL2_Pf2_Errors.npy"), allow_pickle=True)
        elif datatype == "CoH":
            pf2_error = np.load(join(path_here, "sccp/data/CoH_Pf2_Errors.npy"), allow_pickle=True)  
        elif datatype == "scRNA":
            pf2_error = np.load(join(path_here, "sccp/data/scRNA_Pf2_Errors.npy"), allow_pickle=True)

        rank_vec = np.arange(1, len(pf2_error) + 1)
        ax.scatter(rank_vec, pf2_error, c='k', s=20.)
        ax.set(title="R2X", ylabel="Variance Explained", xlabel="Number of Components", ylim=(0, 1), xlim=(0, rank + 0.5), xticks=np.arange(0, rank + 1))
    
    
    
    