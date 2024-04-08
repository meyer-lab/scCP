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
from scipy.stats import linregress, pearsonr, spearmanr

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 16), (6, 1))


    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    cmp=28
    gene = "IFI27"
    dfGene = dfGenePerStatus(X, gene, cellType="leiden")
    dfWP = dfWPPerStatus(X, cmp, cellType="leiden")
    # print(dfGene)
    # print(dfWP)
    # a
    dfWP["Gene"] = dfGene["Average Gene Expression"]
    
    print(dfWP)
    

    idx = len(np.unique(dfWP["Cell Type"]))
    plotCmpPerCellCount(X, dfWP, ax[1:3])
    
    ax[1].set(title=f"WP Cmp:{cmp} V. Gene: {gene}")


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

    df = df.groupby(["Status", "Cell Type", "Condition", "Gene"], observed=True).mean()
    df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

    return df


def dfWPPerStatus(X, cmp, cellType="Cell Type"):
    """Plots average gene expression across cell types for a category of drugs"""
    dataDF = pd.DataFrame(data=X.obsm["weighted_projections"][:, cmp-1], columns=["WP"])
    dataDF["Status"] = X.obs["SLE_status"].values
    dataDF["Condition"] = X.obs["Condition"].values
    dataDF["Cell Type"] = X.obs[cellType].values


    df = pd.melt(
        dataDF, id_vars=["Status", "Cell Type", "Condition"], value_vars="WP"
    ).rename(columns={"variable": "WP", "value": "WProjs"})

    df = df.groupby(["Status", "Cell Type", "Condition", "WP"], observed=True).median().reset_index()

    return df




def plotCmpPerCellCount(X,totaldf, ax):
    """Plot component weights by cell count for a cell type"""

    # totaldf = pd.DataFrame([])
    correlationdf = pd.DataFrame([])


    for i, celltype in enumerate(np.unique(totaldf["Cell Type"])):
        # for j, cond in enumerate(np.unique(df["Condition"])):
        #     status = np.unique(df.loc[df["Condition"] == cond]["Status"])
        #     smalldf = df.loc[(df["Condition"] == cond) & (df["Cell Type"] == celltype)]

        #     if smalldf.empty is False: 
        #         smalldf = smalldf.assign(Cmp=factorsA[j])
        #     else:
        #         smalldf = pd.DataFrame({"Condition": cond, "Cell Type": celltype, "Status": status,
        #                                           cellPerc: 0, "Cmp": factorsA[j]})

            # totaldf = pd.concat([totaldf, smalldf])

        df = totaldf.loc[totaldf["Cell Type"] == celltype]   
        _, _, r_value, _, _ = linregress(df["WProjs"], df["Gene"])
        pearson = pearsonr(df["WProjs"], df["Gene"])[0]
        spearman = spearmanr(df["WProjs"], df["Gene"])[0]
        
        if celltype == "30":    
            sns.scatterplot(data=df, x="WProjs", y="Gene", hue="Status", ax=ax[1])
            ax[1].set(title=f"{celltype}")
        
        correl = [np.round(r_value**2, 3), spearman, pearson]
        test = ["R2 Value ", "Pearson", "Spearman"]
        
        for k in range(3):
            # if k[i] == 0:
            #     k = 
            correlationdf = pd.concat([correlationdf, pd.DataFrame({"Cell Type": str(celltype), "Correlation": [test[k]], "Value": [correl[k]]})])
    
    print(correlationdf)       
    sns.barplot(data=correlationdf.reset_index(), x="Cell Type", y="Value", hue="Correlation", ax=ax[0])
    # ax[1].set_xticks(ax[1].get_xticks())
    # ax[1].set_xticklabels(labels=ax[1].get_xticklabels(), rotation=90)
    # ax[-1].set(title=f"Cmp. {cmp} V. Cell Count")
            
    