"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
import umap
import umap.plot as plt
from .common import subplotLabel, getSetup, umap_axis, flattenData
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import warnings
import seaborn as sns
import matplotlib
from matplotlib import pyplot as mplt
import matplotlib 
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 6), (2, 2))
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
    
    dataDF, _, _ = flattenData(data, factors, projs)
    # allP = np.concatenate(projs, axis=0)
    # CompW = allP @ factors[1]
    # print(np.shape(CompW))

    # UMAP dimension reduction
    # ump = umap.UMAP(random_state=1).fit(allP)
    
    # umapReduc = umap.UMAP(random_state=1)
    pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))
    
    pc = PCA(n_components=rank)
    pcaPoints = pc.fit_transform(data.unfold())
    pcaPoints = umap.UMAP(random_state=1).fit(pcaPoints)


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
    
    
    yup = np.random.choice(a=[False, True], size=np.shape(dataDF)[0], p=[.93, .07])
    
    # drug = "Dexrazoxane HCl (ICRF-187, ADR-529)"
    # drugz = dataDF["Drug"].values
    

    # drugList = np.where(np.asarray(drugz == drug), drug, "Other Drugs")
    
    
    genez = "GNLY"
    geneList = dataDF[genez].to_numpy()
        
    # plt.points(
    #         ump, values=CompW[:, 0], ax=ax[0], show_legend=True, subset_points= yup)
    
    # plt.points(
    #         pf2Points, labels=drugList, ax=ax[0], color_key_cmap='tab20', show_legend=True, subset_points= yup)
        
    # plt.points(
    #         pcaPoints, labels=drugList, ax=ax[1], color_key_cmap='tab20', show_legend=True, subset_points= yup)
    
    # drug = "Triamcinolone Acetonide"
    # drugz = dataDF["Drug"].values
    

    # drugList = np.where(np.asarray(drugz == drug), drug, "Other Drugs")

    
    # plt.points(
    #         pf2Points, labels=drugList, ax=ax[0], color_key_cmap='Paired', show_legend=True, subset_points= yup)
        
    # plt.points(
    #         pcaPoints, labels=drugList, ax=ax[1], color_key_cmap='Paired', show_legend=True, subset_points= yup)

    psm = mplt.pcolormesh([geneList, geneList], cmap=matplotlib.cm.get_cmap('viridis'))


    lb = plt.points(pf2Points, values=geneList, theme='viridis',  background='white',subset_points= yup, ax=ax[0])


    cb = mplt.colorbar(psm, ax=lb)
    
    lb = plt.points(pcaPoints, values=geneList, theme='viridis',  background='white',subset_points= yup, ax=ax[2])

    cb = mplt.colorbar(psm, ax=lb)
    
    
    ax[0].set(
            title="Pf2" + "-Based Decomposition",
        )
    ax[2].set(
            title="PCA" + "-Based Decomposition",
        )
    
    ax[0].set(
        ylabel="UMAP2",
        xlabel="UMAP1")
    ax[1].set(
        ylabel="UMAP2",
        xlabel="UMAP1")
    ax[2].set(
        ylabel="UMAP2",
        xlabel="UMAP1")
    ax[3].set(
        ylabel="UMAP2",
        xlabel="UMAP1")
      
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