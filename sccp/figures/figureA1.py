"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (subplotLabel, getSetup)
import numpy as np
import pandas as pd
import gzip


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (4, 4))

    # Add subplot labels
    subplotLabel(ax)
    
    
    data1 = "/opt/andrew/HamadCITEseq/filtered_feature_bc_matrix_control/barcodes.tsv.gz"
    df1 = pd.read_csv(data1, sep="\t", header=None)
    print(df1)
        # data, sep="\t", header=None, names=("cell_barcode",)
    data2 = "/opt/andrew/HamadCITEseq/filtered_feature_bc_matrix_control/features.tsv.gz"
    df2 = pd.read_csv(data2, sep="\t", header=None)
    print(df2)
    data3 = "/opt/andrew/HamadCITEseq/filtered_feature_bc_matrix_control/matrix.mtx.gz"
    df3 = pd.read_csv(data3, sep="\t", header=None)
    print(df3)

 

    
    
    return f