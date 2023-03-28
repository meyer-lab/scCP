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
    ax, f = getSetup((12, 20), (4, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes(saveXA=False, offset=1.0)
    
    celltype = data["Cell Type"]
    df = celltype.to_dataframe().reset_index()
    df = df.groupby(["Drug", "Cell Type"]).size().reset_index()
    cl = df.columns
    df.columns = [cl[0], cl[1], "Count"]
    
    for i, cell in enumerate(np.unique(celltype)):
        sns.barplot(data=df.loc[df["Cell Type"] == cell], x = "Drug", y = "Count", ax=ax[i])
        ax[i].set_title(cell)
        ax[i].tick_params(axis="x", rotation=90)
    
    return f
