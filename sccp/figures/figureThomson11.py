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
    ax, f = getSetup((3, 3), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    # Import data
    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad", backed="r")
    
    # pf2_batch_df = batch_correction_metrics(X, "projections")
    # pf2_batch_df["Fit"] = "Pf2"
    
    # pc = PCA(n_components=20, svd_solver="arpack")
    # X = X.to_memory()
    # XX = sps.csr_array(X.X)
    # pcaPoints = pc.fit_transform(XX)
    # X.obsm["pca"] = pacmap.PaCMAP().fit_transform(pcaPoints)
    # pca_batch_df = batch_correction_metrics(X, "pca")
    # pca_batch_df["Fit"] = "PCA"
    
    # batch_df =  pd.concat([pf2_batch_df, pca_batch_df])

    # sns.barplot(data=batch_df, x="Value", y="Metric", hue="Fit", ax=ax[0])
    

    return f


def batch_correction_metrics(X, embed):
    """Returns SCIB metrics for batch correction and biological conservation"""
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
                               "S": silh, "ARI": ari, "NMI": nmi}])
    
    metric_df = metric_df.reset_index(drop=True)
    
    metric_df = pd.melt(metric_df, value_vars=["SB", "GC", "iLisi", "cLisi", "ASW", 
                                               "ARI", "S", "NMI"]).rename(
        columns={"variable": "Metric", "value": "Value"})
                                               
    return metric_df