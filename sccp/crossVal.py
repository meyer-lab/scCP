import numpy as np
from .parafac2 import parafac2_nd
from tensorly.parafac2_tensor import parafac2_to_slices


def parafac2_to_tensor(parafac2_tensor):
    _, (A, _, C), projections = parafac2_tensor
    slices = parafac2_to_slices(parafac2_tensor, validate=False)
    lengths = [projection.shape[0] for projection in projections]

    tensor = np.zeros((A.shape[0], max(lengths), C.shape[0]))
    for i, (slice_, length) in enumerate(zip(slices, lengths)):
        tensor[i, :length] = slice_
    return tensor


def shuffle_X(X):
    X = np.copy(X)

    A_idx = np.random.permutation(np.arange(X.shape[0]))
    X = X[A_idx, :, :]

    C_idx = np.random.permutation(np.arange(X.shape[2]))
    X = X[:, :, C_idx]

    for i in range(X.shape[0]):
        B_idx = np.random.permutation(np.arange(X.shape[1]))
        X[i, :, :] = X[i, B_idx, :]

    return X


def crossvalidate(X, rank, trainPerc=0.75, verbose=False):
    X = shuffle_X(X)

    X_B_idx = int(X.shape[1] * trainPerc)
    B_train = X[:, :X_B_idx, :]

    X_C_idx = int(X.shape[2] * trainPerc)
    C_train = X[:, :, :X_C_idx]

    w_B, fac_B, proj_B, _ = parafac2_nd(B_train, rank, verbose=verbose)
    _, _, proj_C, _ = parafac2_nd(C_train, rank, verbose=verbose)

    # Solve procrustes to project C onto B
    proj_B_flat = np.reshape(proj_B, (-1, proj_B.shape[2]))
    proj_C_flat = np.reshape(proj_C[:, :X_B_idx, :], (-1, proj_C.shape[2]))

    S, _, Vh = np.linalg.svd(proj_B_flat.T @ proj_C_flat)
    procM = (S @ Vh).T

    # Project projections into B space
    X_reconst = parafac2_to_tensor((w_B, fac_B, proj_C @ procM))
    X_reconst = np.stack(X_reconst, axis=0)
    X_err = X_reconst - X

    recon_error = np.linalg.norm(X_err[:, X_B_idx:, X_C_idx:]) ** 2
    total_var = np.linalg.norm(X[:, X_B_idx:, X_C_idx:]) ** 2

    return 1.0 - recon_error / total_var


def plotCrossVal(tensor, rank,  ax1, trainPerc=0.75):
    """Creates cross validation accuracy plot for parafac2"""
    rank_vec = np.arange(1, rank + 1)
    cv_error = np.empty(len(rank_vec))

    # Collect Pf2 results
    for i in range(len(rank_vec)):
        cv_err = crossvalidate(tensor, rank=i+1, trainPerc=trainPerc)

    ax1.scatter(rank_vec, cv_error, marker="x", s=20.0)

    ax1.set(
        ylabel="CV Accuracy",
        xlabel="Number of Components",
        xticks=np.arange(0, rank + 1),
        ylim=(-0.05, 1.05),
    )