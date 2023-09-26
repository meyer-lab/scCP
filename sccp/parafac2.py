from typing import Sequence
import numpy as np
from tensorly.tenalg.svd import svd_interface
from tensorly import backend as T


class Pf2X(Sequence):
    def __init__(self, X_list: list, condition_labels, variable_labels):
        assert isinstance(X_list, list)
        self.X_list = X_list
        self.condition_labels = np.array(condition_labels, dtype=object)
        self.variable_labels = np.array(variable_labels, dtype=object)
        assert len(X_list) == len(condition_labels)
        for X in X_list:
            assert X.shape[1] == len(variable_labels)

    def unfold(self):
        return np.concatenate(self.X_list, axis=0)

    def __getitem__(self, item):
        if item >= len(self.X_list):
            raise IndexError("Pf2X index out of range")
        return self.X_list[item]

    def __len__(self):
        return len(self.X_list)


def compressTensor(tensor_slices, compression_threshold=0.0, svd="truncated_svd"):
    """Compress data with the SVD for running Pf2: Takin from Tensorly"""
    loading_matrices = [None for _ in tensor_slices]
    score_matrices = [None for _ in tensor_slices]

    for i, tensor_slice in enumerate(tensor_slices):
        n_rows, n_cols = T.shape(tensor_slice)
        if n_rows <= n_cols and not compression_threshold:
            score_matrices[i] = tensor_slice
            continue

        U, s, Vh = svd_interface(tensor_slice, n_eigenvecs=n_cols, method=svd)

        # Threshold SVD, keeping only singular values that satisfy s_i >= s_0 * epsilon
        # where epsilon is the compression threshold
        num_svds = len([s_i for s_i in s if s_i >= (s[0] * compression_threshold)])
        U, s, Vh = U[:, :num_svds], s[:num_svds], Vh[:num_svds, :]

        # Array broadcasting happens at the last dimension, since Vh is num_svds x n_cols
        # we need to transpose it, multiply in the singular values and then transpose
        # it again. This is equivalen to writing diag(s) @ Vh. If we skip the
        # transposes, we would get Vh @ diag(s), which is wrong.
        score_matrices[i] = T.transpose(s * T.transpose(Vh))
        loading_matrices[i] = U

    return score_matrices
