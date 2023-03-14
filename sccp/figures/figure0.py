"""
Creating synthetic data and implementation of parafac2
"""
import numpy as np
import xarray as xa
from .common import (
    subplotLabel,
    getSetup,
    plotFactors,
    plotProj,
    plotSS,
)
from ..synthetic import synthXA, plot_synth_pic
from ..parafac2 import parafac2_nd
from ..decomposition import plotR2X
from ..crossVal import plotCrossVal


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    blobInfo, blobDF = synthXA(magnitude=20, type="beach")

    # Performing parafac2 on single-cell Xarray
    _, factors, projs, _ = parafac2_nd(
        blobInfo["data"].to_numpy(),
        rank=2,
        verbose=True,
    )

    plotFactors(factors, blobInfo["data"], ax)

    projs = xa.DataArray(
        projs,
        dims=["Time", "Cell", "Cmp"],
        coords=dict(
            Time=blobInfo.coords["Time"],
            Cell=blobInfo.coords["Cell"],
            Cmp=[f"Cmp. {i}" for i in np.arange(1, projs.shape[2] + 1)],
        ),
        name="projections",
    )
    projs = xa.merge([projs, blobInfo["Cell Type"]], compat="no_conflicts")

    flattened_projs = projs.stack(AllCells=("Time", "Cell"))

    # Remove empty slots
    nonzero_index = np.any(flattened_projs["projections"].to_numpy() != 0, axis=0)
    flattened_projs = flattened_projs.isel(AllCells=nonzero_index)

    gini_coeff(flattened_projs)
    # plotSS(flattened_projs, ax[3])

    # idxx = np.random.choice(
    #     len(flattened_projs.coords["AllCells"]), size=200, replace=False
    # )
    # plotProj(flattened_projs.isel(AllCells=idxx), ax[4:6])

    # plotR2X(blobInfo["data"].to_numpy(), 3, ax[7])
    # plotCrossVal(blobInfo["data"].to_numpy(), 3,  ax[8], trainPerc=0.75)
    
    pj = flattened_projs["projections"].to_numpy()
    array = np.sort(pj[1, :])
    if np.amin(array) < 0:
        array -= np.amin(array)
    index = np.arange(1,array.shape[0]+1)
    n = array.shape[0]
    gini = ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) 
    
    # print(gini)
    # array = array.flatten() #all values are treated equally, arrays must be 1d
    # if np.amin(array) < 0:
    #     array -= np.amin(array) #values cannot be negative
    # array += 0.0000001 #values cannot be 0
    # array = np.sort(array) #values must be sorted
    # index = np.arange(1,array.shape[0]+1) #index per array element
    # n = array.shape[0]#number of array elements
    # return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) 
    
    
  
    return f

def gini_coeff(projs: xa.Dataset):
    proj_data = projs["projections"].to_numpy()
    assert proj_data.ndim == 2
    
    gini = np.empty(proj_data.shape[0])
    for i in range(proj_data.shape[0]):
        print(i)
        projComp = np.sort(proj_data[i, :])
        if np.amin(projComp) < 0:
            projComp -= np.amin(projComp)
        index = np.arange(1, projComp.shape[0]+1)
    
        gini[i] = ((np.sum((2 * index - projComp.shape[0]  - 1) * projComp)) / 
                projComp.shape[0] * np.sum(projComp))
        
    idx_gini = np.argsort(gini)
    
    print(idx_gini)
    print(gini)
        
    return
        
        


palette = {
    "Ground": "khaki",
    "Leaf1": "limegreen",
    "Leaf2": "darkgreen",
    "Sun": "yellow",
    "Trunk1": "sienna",
    "Trunk2": "chocolate",
}
color_palette = ["khaki", "limegreen", "darkgreen", "yellow", "sienna", "chocolate"]
