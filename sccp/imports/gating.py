import numpy.typing as npt
import pandas as pd


def gateThomsonCells(X) -> npt.ArrayLike:
    """Manually gates cell types for Thomson UMAP"""
    umap1 = X.obsm["embedding"][:, 0]
    umap2 = X.obsm["embedding"][:, 1]

    df = pd.DataFrame(data={"UMAP1": umap1, "UMAP2": umap2})
    df["Cell Type"] = "Monocytes"

    idx = df.index[(df["UMAP1"] >= 5)].tolist()
    df.loc[idx, "Cell Type"] = "DCs"

    idx = df.index[(df["UMAP2"] >= 9.5)].tolist()
    df.loc[idx, "Cell Type"] = "B Cells"

    idx = df.index[
        (df["UMAP1"] >= -5)
        & (df["UMAP1"] <= 5)
        & (df["UMAP2"] >= -3)
        & (df["UMAP2"] <= 5)
    ].tolist()
    df.loc[idx, "Cell Type"] = "Monocytes"

    idx = df.index[
        (df["UMAP1"] <= -0.75) & (df["UMAP2"] >= 5) & (df["UMAP2"] <= 9.5)
    ].tolist()
    df.loc[idx, "Cell Type"] = "NK Cells"

    idx = df.index[
        (df["UMAP1"] >= -0.75) & (df["UMAP2"] >= 5) & (df["UMAP2"] <= 9.5)
    ].tolist()
    df.loc[idx, "Cell Type"] = "T Cells"

    return df["Cell Type"].values
