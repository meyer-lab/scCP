"""
Thomson: Further examination of cells based on their components
"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import scib
import scanpy as sc
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import pacmap
import numpy as np
import scipy.sparse as sps


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((3, 3), (1, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import data
    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad", backed="r")
    
    
    
    # plot_batch_correction(X, "projections", ax[0])
    
    # pc = PCA(n_components=20, svd_solver="arpack")
    # X = X.to_memory()
    # XX = sps.csr_array(X.X)
    # pcaPoints = pc.fit_transform(XX)

    # X.obsm["pca"] = pacmap.PaCMAP().fit_transform(pcaPoints)
    # plot_batch_correction(X, "pca", ax[1])
    
    
  
    return f

def plot_batch_correction(X, embed, ax):
    clisi = scib.me.clisi_graph(X.to_memory(), label_key="Cell Type", type_="embed", use_rep=embed)
    asw = scib.me.isolated_labels_asw(X.to_memory(), batch_key="Condition", label_key="Cell Type", embed=embed)
    silh = scib.me.silhouette(X, label_key="Cell Type", embed=embed)
    sb = scib.me.silhouette_batch(X, batch_key="Condition", label_key="Cell Type", embed=embed)
    ilisi = scib.me.ilisi_graph(X.to_memory(), batch_key="Condition", type_="embed", use_rep=embed)
    sc.pp.neighbors(X, use_rep=embed)
    scib.me.isolated_labels_f1(X.to_memory(), batch_key="Condition", label_key="Cell Type", embed=embed)
    gc = scib.me.graph_connectivity(X, label_key="Cell Type")
    scib.me.cluster_optimal_resolution(X, cluster_key="leiden", label_key="Cell Type")
    ari = scib.me.ari(X, cluster_key="leiden", label_key="Cell Type")
    nmi = scib.me.nmi(X, cluster_key="leiden", label_key="Cell Type")

    
    metric_df = pd.DataFrame([{"SB": sb, "GC": gc, "iLisi": ilisi, "cLisi": clisi, "ASW": asw,
                               "Silhouette": silh, "ARI": ari, "NMI": nmi}])
    
    metric_df = metric_df.reset_index(drop=True)
    
    metric_df = pd.melt(metric_df, value_vars=["SB", "GC", "iLisi", "cLisi", "ASW", 
                                               "ARI", "Silhouette", "NMI"]).rename(
        columns={"variable": "Metric", "value": "Value"})
    
    sns.barplot(data=metric_df, x="Metric", y="Value", ax=ax)
    
