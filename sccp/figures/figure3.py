"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import (
    subplotLabel,
    getSetup,
)
from ..imports.scRNA import ThompsonXA_SCGenes


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 20), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes(saveXA=False, offset=1.0)
    
    celltype = data["Cell Type"]
    
    df = celltype.to_dataframe().reset_index()
    df = df[df["Cell Type"] != ""]
    cellCount = df.groupby(["Drug"]).size().values
    df = df.groupby(["Drug", "Cell Type"]).size().unstack(fill_value=0).stack().reset_index()
    
    cl = df.columns
    df.columns = [cl[0], cl[1], "Count"]

    for i, cell in enumerate(np.unique(df["Cell Type"])):
        data = df.loc[df["Cell Type"] == cell]
        data.loc[:, "Count"] /= cellCount
        sns.barplot(data=data, x = "Drug", y = "Count", ax=ax[i])
        ax[i].set_title(cell)
        ax[i].tick_params(axis="x", rotation=90)
    
    return f
