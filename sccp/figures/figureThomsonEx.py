"""
Thomson: Plotting Pf2 factors and weights
"""
from .common import subplotLabel, getSetup, openPf2
from tlviz.factor_tools import factor_match_score as fms
from ..parafac2 import pf2
from ..imports import import_thomson
from tensorly.cp_tensor import cp_flip_sign, CPTensor


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 12), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    rank = 2
    X = import_thomson()
    X = pf2(X, rank, random_state=1, doEmbedding=False)
    
    Xcp = CPTensor((X.uns["Pf2_weights"], [X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]]))

    XX = pf2(X, rank, random_state=3, doEmbedding=False)
    XXcp = CPTensor((XX.uns["Pf2_weights"], [XX.uns["Pf2_A"], XX.uns["Pf2_B"], XX.varm["Pf2_C"]]))
    
    fmsScore = fms(Xcp, XXcp, consider_weights=True, skip_mode=None)
    print(fmsScore)
  
    return f

