"""
Lupus
"""
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
import pandas as pd
from .figureCITEseq5 import top_bot_genes
from scipy.stats import linregress

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 16), (5, 4))


    # Add subplot labels
    subplotLabel(ax)
    
    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    cmp=28
    genes = top_bot_genes(X, cmp=cmp, geneAmount=5)
    print(genes)

    geneDF = dfGenePerStatus(X, genes[0], cellType="Cell Type2")
    idx = len(np.unique(geneDF["Cell Type"]))
    
    plotCmpPerGene(X, cmp, geneDF, ax[0:idx])
    
    return f



def dfGenePerStatus(X, gene, cellType="Cell Type"):
    """Plots average gene expression across cell types for a category of drugs"""
    genesV = X[:, gene]
    dataDF = genesV.to_df()
    dataDF = dataDF.subtract(genesV.var["means"].values)
    dataDF["Status"] = genesV.obs["SLE_status"].values
    dataDF["Condition"] = genesV.obs["Condition"].values
    dataDF["Cell Type"] = genesV.obs[cellType].values

    df = pd.melt(
        dataDF, id_vars=["Status", "Cell Type", "Condition"], value_vars=gene
    ).rename(columns={"variable": "Gene", "value": "Value"})

    df = df.groupby(["Status", "Cell Type", "Gene", "Condition"], observed=True).mean()
    df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

    return df

   
def plotCmpPerGene(X, cmp, geneDF, ax):
    yt = np.unique(X.obs["Condition"])
    factorsA = np.array(X.uns["Pf2_A"])
    factorsA = factorsA[:, cmp-1]

    totaldf = pd.DataFrame([])
    gene = np.unique(geneDF["Gene"].to_numpy())[0]
    geneDF["Condition"] = pd.Categorical(geneDF["Condition"], yt)
    
    for i, celltype in enumerate(np.unique(geneDF["Cell Type"])):
        for j, cond in enumerate(np.unique(geneDF["Condition"])):
            smalldf = geneDF.loc[(geneDF["Condition"] == cond) & (geneDF["Cell Type"] == celltype)]
        
            if smalldf.empty is False: 
                smalldf = smalldf.assign(Cmp=factorsA[j])

            totaldf = pd.concat([totaldf, smalldf])
         
        df = totaldf.loc[totaldf["Cell Type"] == celltype]
        _, _, r_value, _, _ = linregress(df["Cmp"], df["Average Gene Expression"])
        sns.scatterplot(data=df, x="Cmp", y="Average Gene Expression", hue="Status", ax=ax[i])
        ax[i].set(title=f"{celltype}: R2 Value - {np.round(r_value**2, 3)}", xlabel=f"Cmp. {cmp}", ylabel=f"Average Gene Expression: {gene}")