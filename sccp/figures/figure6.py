"""
XXX
"""

import anndata
import numpy as np
import pacmap
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
from .common import getSetup, subplotLabel
import seaborn as sns
import pandas as pd

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((18, 8), (4, 10))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/pf2/thomson_fitted.h5ad")
    n_components = 20
    
    pc = PCA(n_components=n_components)
    pca = pc.fit(np.asarray(X.X - X.var["means"].values))

    # gene_corr_matrix = np.zeros((n_components, n_components))
    # for i in range(n_components):
    #     for j in range(n_components):
    #         corr, _ = spearmanr(pca.components_.T[:, i], X.varm["Pf2_C"][:, j])
    #         gene_corr_matrix[i, j] = corr
            
    # mask = np.triu(np.ones_like(gene_corr_matrix, dtype=bool))

    # for i in range(len(mask)):
    #     mask[i, i] = False
    # ticks = np.arange(1, gene_corr_matrix.shape[1]+1)
    # print(ticks)
    # sns.heatmap(
    #     data=gene_corr_matrix,
    #     # vmin=0.5,
    #     # vmax=1,
    #     center=0,
    #     cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True),
    #     xticklabels=ticks,
    #     yticklabels=ticks,
    #     mask=mask,
    #     cbar_kws={"label": "Prediction Accuracy"},
    #     ax=ax[0])
    
    
    
    for i in range(1, n_components):
        plot_gene_factors_partial(i, X, pca, ax[(2*i)-2], geneAmount=10, top=True)
        plot_gene_factors_partial(i, X, pca, ax[(2*i)+1-2], geneAmount=10, top=False)
    return f


def plot_gene_factors_partial(
    cmp: int, X, pca, ax, geneAmount: int = 5, top=True
):
    """Plotting weights for gene factors for both most negatively/positively weighted terms"""
    cmpName = f"Cmp. {cmp}"

    df = pd.DataFrame(
        data=pca.components_.T[:, cmp - 1], index=X.var_names, columns=[cmpName]
    )
    df = df.reset_index(names="Gene")
    df = df.sort_values(by=cmpName)

    if top:
        sns.barplot(
            data=df.iloc[-geneAmount:, :], x="Gene", y=cmpName, color="k", ax=ax
        )
    else:
        sns.barplot(data=df.iloc[:geneAmount, :], x="Gene", y=cmpName, color="k", ax=ax)

    ax.tick_params(axis="x", rotation=90)