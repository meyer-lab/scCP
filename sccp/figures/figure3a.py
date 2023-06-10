"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
import umap
import umap.plot as plt
from .common import subplotLabel, getSetup, umap_axis
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import warnings
import seaborn as sns
import matplotlib
from matplotlib import pyplot as mplt

warnings.filterwarnings("ignore")


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 10), (2, 2))
    subplotLabel(ax)  # Add subplot labels

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    rank = 25
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
        verbose=True,
    )
    allP = np.concatenate(projs, axis=0)
    CompW = allP @ factors[1]
    print(np.shape(CompW))

    # UMAP dimension reduction
    ump = umap.UMAP(random_state=1).fit(allP)


    # plt.points(
    #         ump, values=CompW[:, 2], ax=ax[2], width=400, height=400
    #     )
    # plt.points(
    #         ump, values=CompW[:, 1], ax=ax[1], width=200, height=200
        # )
   
    # ax[2].legend()
        # umap_axis(x, y, ax[i])
        
        
    cmap = "viridis"
    # create a scalar colour map for values
    # norm = matplotlib.colors.Normalize(CompW[:, 0].min(), CompW[:, 0].max())
    # scalar_map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)  # type: ignore
    # # plot using umaps helper function
    # # plot.points(mapper, values=values, ax=ax, cmap=cmap)
    # # create a colorbar
    # cbar = f.colorbar(scalar_map, ax=ax[0])  # type: ignore
    
    
    yup = np.random.choice(a=[False, True], size=np.shape(CompW)[0], p=[.9, .1])
    
    plt.points(
            ump, values=CompW[:, 0], ax=ax[0], show_legend=True, subset_points= yup)
        # )
      
    # plt.points(
    #         ump, values=CompW[:, 0], ax=ax[1], cmap=cmap
    #     )


    # ax[0].set(title=f"Component {0 + 1}")

    # ax[1].set(title=f"Component {1 + 1}")


    # #define color gradient for measurements, I did it with seaborn
    # gradient = sns.color_palette("Blues_r", as_cmap = True)

    # #normalise the measurement that should represent your color data
    # normalised_perimeter = plt.Normalize(CompW[:, 0].min(), CompW[:, 0].max())

    # #define a colorbar that is added to the plot
    # colorbar = plt.cm.ScalarMappable(norm = normalised_perimeter, cmap = gradient)

    # #set the min, max values of the colorbar
    # colorbar.set_array(CompW[:, 0])

    return f

# https://lightrun.com/answers/lmcinnes-umap-pltcolorbar-support-for-umapplotpoints