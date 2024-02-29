"""
Thomson: Plotting normalized genes and separating data by status (and celltype)
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
from .commonFuncs.plotUMAP import plotCmpGeneWeightedUMAP, plotCmpGeneWeightedPerCellType

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 18), (6, 4))
    ax, f = getSetup((12, 6), (2, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    # comps = [22, 28]
    
    # total_df = pd.DataFrame([])
    
    # cmpWeights = np.concatenate(([X.varm["Pf2_C"][:, comps[0]-1]], [X.varm["Pf2_C"][:, comps[1]-1]]))

    # df = pd.DataFrame(data=cmpWeights.transpose(), index=X.var_names, columns=["Component 22", "Component 28"])

    # sns.scatterplot(data=df, x="Component 22", y="Component 28", ax=ax[0])
    
    # print(df.sort_values(by=["Component 22"]))
    # print(df.sort_values(by=["Component 28"]))
    
    # print(df.iloc[0:-2,:])
    # newdf = df.sort_values(by=["Component 28"])
    # sns.scatterplot(data=newdf.iloc[:-1,:], x="Component 22", y="Component 28", ax=ax[1])
    # ax[0].set(xscale="log", yscale="log")
    


 
    
    
    
    genes1 = top_bot_genes(X, cmp=22, geneAmount=40)
    
    for i in range(len(genes1)):
        print(genes1[i])
    # genes1 = top_bot_genes(X, cmp=comps[0], geneAmount=6)
    

    
    
    # # comps = [8, 9, 10, 13, 22, 28, 4, 14]
    # plotCmpGeneWeightedUMAP(X, 4, ax[0], cbarMax=.4)
    # plotCmpGeneWeightedPerCellType(X, 4, ax[1], cellType="Cell Type2")
    
    # plotCmpGeneWeightedUMAP(X, 14, ax[2], cbarMax=.4)
    # plotCmpGeneWeightedPerCellType(X, 14, ax[3], cellType="Cell Type2")


    # # Cmp.4 Most weighted pos/neg genes
    # genes = [
    #     ["GZMK", "DUSP2"],
    #     ["CMC1", "LYAR"],
    #     ["AC092580.4", "CD8B"],
    #     ["FAM173A", "CLDND1"],
    #     ["PIK3R1", "CD8A"],
    #     ["SPON2", "FGFBP2"],
    #     ["GZMB", "PRF1"],
    #     ["GZMH", "CLIC3"],
    #     ["NKG7", "GNLY"],
    #     ["CCL4", "CD247"],
    # ]
    # comps = [8, 9, 10, 13, 22, 28, 4, 14]
    # genes = top_bot_genes(X, cmp=4, geneAmount=6)

    # for i, gene in enumerate(np.ravel(genes)):
    #     plotGenePerStatus(X, gene, ax[i], cellType="Cell Type2")
    #     ax[i].set_xticklabels(labels=ax[i].get_xticklabels(), rotation=90)

    # genes = [["CLEC4E", "RETN"]]

    # for i, gene in enumerate(genes):
    #     plot2GenePerCellTypeStatus(X, genes[i], genes[-(1+i)], "NK Dim", "T8 GZMH", ax[i], cellType="Cell Type2")
    
  
    # plot2GenePerCellTypeStatus(X, "APOBEC3A", "BANK1", "CM", "B Naive", ax[0], cellType="Cell Type2")


    return f



def plotGenePerStatus(X, gene, ax, cellType="Cell Type"):
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

    df = df.groupby(["Status", "Cell Type", "Gene", "Condition"], observed=False).mean()
    df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()
    
    

    sns.boxplot(
        data=df.loc[df["Gene"] == gene],
        x="Cell Type",
        y="Average Gene Expression",
        hue="Status",
        ax=ax,
        showfliers=False,
    )
    ax.set(title=gene)



def plot2GenePerCellTypeStatus(
    X, gene1, gene2, celltype1, celltype2, ax, cellType="Cell Type"
):
    """Plots average gene expression across cell types for a category of drugs"""
    gene = [gene1, gene2]

    for i in range(2):
        genesV = X[:, gene[i]]
        dataDF = genesV.to_df()
        dataDF = dataDF.subtract(genesV.var["means"].values)
        dataDF["Status"] = genesV.obs["SLE_status"].values
        dataDF["Condition"] = genesV.obs["Condition"].values
        dataDF["Cell Type"] = genesV.obs[cellType].values

        df = pd.melt(
            dataDF, id_vars=["Status", "Cell Type", "Condition"], value_vars=gene[i]
        ).rename(columns={"variable": "Gene", "value": "Value"})

        df = df.groupby(
            ["Status", "Cell Type", "Gene", "Condition"], observed=False
        ).mean()
        df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

        if i == 0:
            df_ = df.copy()
        else:
            df = pd.concat([df_, df]).reset_index()

    df = df.pivot(
        index=["Status", "Cell Type", "Condition"],
        columns="Gene",
        values="Average Gene Expression",
    ).reset_index()
    df1 = df.loc[df["Cell Type"] == celltype1]
    df2 = df.loc[df["Cell Type"] == celltype2]
    df = pd.concat([df1, df2]).reset_index()
    
    df = df.dropna(subset=gene[0])


    sns.scatterplot(
        data=df,
        x=gene[0],
        y=gene[1],
        hue="Status",
        ax=ax,
    )

    ax.set_title("Average Gene Expression Per Patient")
    ax.set_xlabel(f"{celltype1}: {gene1}")
    ax.set_ylabel(f"{celltype2}: {gene2}")
