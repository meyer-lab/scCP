import pandas as pd
from ..figures.common import openPf2


def gateThomsonCells():
    """Manually gates cell types for Thomson UMAP"""
    X = openPf2(30, "Thomson")

    umap1 = pf2Points[:, 0]
    umap2 = pf2Points[:, 1]

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
