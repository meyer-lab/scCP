"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
import numpy as np
import xarray as xa
import pandas as pd
import seaborn as sns
from .common import (
    subplotLabel,
    getSetup,
    plotFactors,
    plotProj,
    plotSS,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
from ..decomposition import plotR2X
from ..crossVal import plotCrossVal


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes(saveXA=False, offset=1.0)
    
    drugXA = data.sel(Drug=["Budesonide", "CTRL2"], Gene= ["CD163", "ADORA3", "MS4A6A"])
    df = drugXA.to_dataframe().reset_index()
    
    sns.violinplot(data=df.loc[df["Gene"] == "ADORA3"] , x="Cell Type", y="data", hue="Drug",
               linewidth=1,split=True, ax=ax[0])
    
    sns.violinplot(data=df.loc[df["Gene"] == "CD163"] , x="Cell Type", y="data", hue="Drug",
               linewidth=1,split=True, ax=ax[1])
    
    sns.violinplot(data=df.loc[df["Gene"] == "MS4A6A"] , x="Cell Type", y="data", hue="Drug",
               linewidth=1, split=True, ax=ax[2])
    
    drugXA = data.sel(Drug=["Dexrazoxane HCl (ICRF-187, ADR-529)", "CTRL6"], Gene= ["CD163", "ADORA3", "MS4A6A"])
    df = drugXA.to_dataframe().reset_index()
    
    sns.violinplot(data=df.loc[df["Gene"] == "ADORA3"] , x="Cell Type", y="data", hue="Drug",
               linewidth=1,split=True, ax=ax[3])
    
    sns.violinplot(data=df.loc[df["Gene"] == "CD163"] , x="Cell Type", y="data", hue="Drug",
               linewidth=1,split=True, ax=ax[4])

    sns.violinplot(data=df.loc[df["Gene"] == "MS4A6A"] , x="Cell Type", y="data", hue="Drug",
               linewidth=1, split=True, ax=ax[5])
    
    drugXA = data.sel(Drug=["Flurbiprofen", "CTRL6"], Gene= ["LTA", "IL2RA", "CD40LG"])
    df = drugXA.to_dataframe().reset_index()
    
    sns.violinplot(data=df.loc[df["Gene"] == "LTA"] , x="Cell Type", y="data", hue="Drug",
               linewidth=1,split=True, ax=ax[6])
    
    sns.violinplot(data=df.loc[df["Gene"] == "IL2RA"] , x="Cell Type", y="data", hue="Drug",
               linewidth=1,split=True, ax=ax[7])

    sns.violinplot(data=df.loc[df["Gene"] == "CD40LG"] , x="Cell Type", y="data", hue="Drug",
               linewidth=1, split=True, ax=ax[8])
    
    
    genes = ["CD163", "ADORA3", "MS4A6A", "CD163", "ADORA3", "MS4A6A", "LTA", "IL2RA", "CD40LG"]
    for i in range(9):
        ax[i].tick_params(axis="x", rotation=45)
        ax[i].set_title("Gene:"+ genes[i])
    
    return f
