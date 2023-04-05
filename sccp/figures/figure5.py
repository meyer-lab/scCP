"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
import seaborn as sns
import xarray as xa
import pandas as pd
from .common import (
    subplotLabel,
    getSetup,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import umap 



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 25), (2, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes(offset=1.0)

    _, factors, projs, _ = parafac2_nd(
        data,
        rank=13,
    )

    flatProjs = np.concatenate(projs, axis=0)
    idxx = np.random.choice(flatProjs.shape[0], size=200, replace=False)
    flatProjs = flattened_projs[idxx, :]
    
    return f

# def makeFigure():
#     """Get a list of the axis objects and create a figure."""
#     # Get list of axis objects
#     ax, f = getSetup((12, 12), (3, 3))

#     # Add subplot labels
#     subplotLabel(ax)

#     # Import of single cells: [Drug, Cell, Gene]
#     data = ThompsonXA_SCGenes(saveXA=False, offset=1.0)

#     # Performing parafac2 on single-cell Xarray
#     _, factors, projs, _ = parafac2_nd(
#         data["data"].to_numpy(),
#         rank=13,
#     )

#     projs = xa.DataArray(
#         projs,
#         dims=["Drug", "Cell", "Cmp"],
#         coords=dict(
#             Drug=data.coords["Drug"],
#             Cell=data.coords["Cell"],
#             Cmp=[f"Cmp. {i}" for i in np.arange(1, projs.shape[2] + 1)],
#         ),
#         name="projections",
#     )
#     projs = xa.merge([projs, data["Cell Type"]], compat="no_conflicts")

#     flattened_projs = projs.stack(AllCells=("Drug", "Cell"))
#     flat_data = data["data"].stack(AllCells=(("Drug", "Cell")))
  
#     # Remove empty slots
#     nonzero_index = np.any(flattened_projs["projections"].to_numpy() != 0, axis=0)
#     flattened_projs = flattened_projs.isel(AllCells=nonzero_index) 
#     flat_data = flat_data.isel(AllCells=nonzero_index) 
    
#     # UMAP dimension reduction
#     umapReduc = umap.UMAP()
#     umapPoints = umapReduc.fit_transform(flattened_projs["projections"].to_numpy().T)
    
#     # Find cells associated with drugs
#     drugs = ["Triamcinolone Acetonide", "Budesonide", "Betamethosone Valerate", "Dexrazoxane HCl (ICRF-187, ADR-529)"]
    
#     for i, drugz in enumerate(drugs):
#         flatData = flat_data.where(flat_data.Drug == drugz).to_numpy()
#         flatData = np.nan_to_num(flatData[0, :], nan=-1)
#         flatData[flatData != -1] = 1

#         # Creating DF for figure
#         umapDF = pd.DataFrame({"UMAP1": umapPoints[:, 0],
#                 "UMAP2": umapPoints[:, 1],
#                 drugz: flatData,
#             })
#         sns.scatterplot(data=umapDF, x="UMAP1", y="UMAP2",hue=drugz, s=0.5, ax=ax[i])
        
    
#     genes = ["LAD1", "NKG7", "CD8A", "CD33", "CCR7"]
#     # DC, NK, CD8, Mono, CD4]
#     for i, genez in enumerate(genes):
#         gene_values = flat_data.sel(Gene=genez).to_numpy()
#         tl = ax[i+4].scatter(umapPoints[::25, 0], umapPoints[::25, 1], c=gene_values[::25], cmap ="cool", s=0.5)
#         f.colorbar(tl, ax=ax[i+4])
#         ax[i+4].set_xlabel("UMAP1")
#         ax[i+4].set_ylabel("UMAP2")
#         ax[i+4].set_title(genez)

#     return f
