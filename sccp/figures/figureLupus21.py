"""
Lupus
"""
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
import pandas as pd
from .figureCITEseq5 import top_bot_genes
from scipy.stats import linregress
from .figureLupus19 import cmpGatedDF
from .figureLupus17 import dfGenePerStatus, plotCmpPerGene
import scanpy as sc


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 16), (5, 4))


    # Add subplot labels
    subplotLabel(ax)
    
    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    df = pd.read_csv("LupusGeneScoresCD8.csv")
    a = df.loc[df["Module"] == "Cytotoxic"]["Gene"].values
    print(np.shape(a))

    print(sc.tl.score_genes(adata=X, gene_list=df.loc[df["Module"] == "Cytotoxic"]["Gene"].values))
    # print(a)



    return f

