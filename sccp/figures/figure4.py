# """
# Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
# """
# import numpy as np
# from .common import (
#     subplotLabel,
#     getSetup,
#     plotFactors,
#     plotProj,
# )
# from ..imports.scRNA import ThompsonXA_SCGenes
# from ..parafac2 import parafac2_nd
# from ..decomposition import plotR2X
# from ..crossVal import plotCrossVal



# def makeFigure():
#     """Get a list of the axis objects and create a figure."""
#     # Get list of axis objects
#     ax, f = getSetup((9, 8), (2, 2))
#     # ax, f = getSetup((9, 12), (2, 2))

#     # Add subplot labels
#     subplotLabel(ax)

#     # Import of single cells: [Drug, Cell, Gene]
#     data = ThompsonXA_SCGenes()

#     rank = 25
#     _, factors, projs, _ = parafac2_nd(
#         data,
#         rank=rank,
#         random_state=1,
#     )

#     # flattened_projs = np.concatenate(projs, axis=0)
#     # idxx = np.random.choice(flattened_projs.shape[0], size=200, replace=False)

#     # plotFactors(factors, data, ax[0:3], reorder=(0, 2), trim=(2,))

#     # plotProj(flattened_projs[idxx, :], ax[3:5])

#     # plotR2X(data, 30, ax[2])

#     # plotCrossVal(data.X_list, 3, ax[3], trainPerc=0.75)
    
#     # ax[1].set(ylabel=None)
#     # ax[2].axes.yaxis.set_ticklabels([])

#     return f


"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
import numpy as np
from .common import (
    subplotLabel,
    getSetup,
    plotFactors,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
from ..decomposition import R2XRank
from ..crossVal import CrossValRank
import pandas as pd
import seaborn as sns



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 10), (2, 2))
    # ax, f = getSetup((9, 12), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()

    rank = 5
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=0,
    )

    # flattened_projs = np.concatenate(projs, axis=0)
    # idxx = np.random.choice(flattened_projs.shape[0], size=200, replace=False)

    # plotFactors(factors, data, ax[0:3], reorder=(0, 2), trim=(2,))

    # plotProj(flattened_projs[idxx, :], ax[3:5])

    r2x_error = R2XRank(data, rank+3)

    cv_error = CrossValRank(data.X_list, rank+3, trainPerc=0.75)
    
    error = np.concatenate((r2x_error[0], r2x_error[1], cv_error[0], cv_error[1]))
    errorDecomp= ["Pf2", "PCA"]
    errorType = ["Fit", "Cross Validation"]
    
    df = pd.DataFrame([])
    iter = 0
    for j in range(len(errorType)):
        for k in range(len(errorDecomp)):
            for i in range(rank):
                df = pd.concat([df, pd.DataFrame({"Variance Explained": error[iter], "Component": [i+1], "Decomposition": errorDecomp[k],
                                                           "Fitting Type": errorType[j]})])
                iter+=1
                
    print(df)
    
    
    sns.scatterplot(data=df, x="Component", y="Variance Explained", hue="Fitting Type", style="Decomposition", ax=ax[0])
    ax[0].set(
        xticks=np.arange(0, rank + 1, 2),
        ylim=(0, np.max(error) + 0.005),
    )
    # dataDF = pd.DataFrame(data=error, columns=)
    
    # ax[1].set(ylabel=None)
    # ax[2].axes.yaxis.set_ticklabels([])

    return f


