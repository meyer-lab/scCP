import umap
import numpy as np
import pandas as pd
import umap.plot
from ..figures.common import openPf2


def gateThomsonCells(rank):
    """Manually gates cell types for Thomson UMAP"""
    _, _, projs = openPf2(rank, "Thomson")

    pf2Points = umap.UMAP(random_state=1).fit_transform(projs)

    df = pd.DataFrame(
        data={
            "UMAP1": pf2Points[:, 0],
            "UMAP2": pf2Points[:, 1],
            "Cell Type": np.zeros(pf2Points.shape[0]),
        }
    )

    df.loc[df["UMAP1"] >= 5, "Cell Type"] = "DCs"
    df.loc[df["UMAP2"] >= 9.5, "Cell Type"] = "B Cells"

    idx = (
        (df["UMAP1"] >= -5)
        & (df["UMAP1"] <= 5)
        & (df["UMAP2"] >= -3)
        & (df["UMAP2"] <= 5)
    )
    df.loc[idx, "Cell Type"] = "Monocytes"

    idx = (df["UMAP1"] <= -0.75) & (df["UMAP2"] >= 5) & (df["UMAP2"] <= 9.5)
    df.loc[idx, "Cell Type"] = "NK Cells"

    idx = (df["UMAP1"] >= -0.75) & (df["UMAP2"] >= 5) & (df["UMAP2"] <= 9.5)
    df.loc[idx, "Cell Type"] = "T Cells"

    return df["Cell Type"]
