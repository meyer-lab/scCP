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
    ax, f = getSetup((5, 5), (2, 2))


    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    cmp=28
    gene = "IFI27"
    dfGene = dfGenePerStatus(X, gene, cellType="leiden")
    dfWP = dfWPPerStatus(X, cmp, cellType="leiden")
    dfWP["Gene"] = dfGene["Average Gene Expression"]

    
    # dfWP = dfWP.replace("44", "Non")
    louvainclusters = np.unique(dfWP["Cell Type"])
    result = np.where(louvainclusters=="30")[0]
    louvainclusters = np.delete(louvainclusters,result)
    
    dfWP = dfWP.replace(louvainclusters, "Non")
    print(np.unique(dfWP["Cell Type"]))
    print(dfWP)

    sns.scatterplot(dfWP, x="WProjs", y="Gene", hue="Cell Type", ax=ax[0])
    ax[0].set_yscale("log")


    return f

def dfGenePerStatus(X, gene, cellType="Cell Type"):
    """Plots average gene expression across cell types for a category of drugs"""
    genesV = X[:, gene]
    dataDF = genesV.to_df()
    # dataDF = dataDF.subtract(genesV.var["means"].values)
    dataDF["Status"] = genesV.obs["SLE_status"].values
    dataDF["Condition"] = genesV.obs["Condition"].values
    dataDF["Cell Type"] = genesV.obs[cellType].values


    df = pd.melt(
        dataDF, id_vars=["Status", "Cell Type", "Condition"], value_vars=gene
    ).rename(columns={"variable": "Gene", "value": "Value"})

    df = df.groupby(["Status", "Cell Type", "Gene", "Condition"], observed=True).mean()
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

    df = df.groupby(["Status", "Cell Type", "WP", "Condition"], observed=True).mean().reset_index()

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

        # sns.scatterplot(data=df, x="WProjs", y="Gene", hue="Status", ax=ax[i])
        # ax[i].set(title=f"{celltype}: R2 Value - {np.round(r_value**2, 3)}")
        
        correl = [np.round(r_value**2, 3), spearman, pearson]
        test = ["R2 Value ", "Pearson", "Spearman"]
        
        for k in range(3):
            # if k[i] == 0:
            #     k = 
            correlationdf = pd.concat([correlationdf, pd.DataFrame({"Cell Type": str(celltype), "Correlation": [test[k]], "Value": [correl[k]]})])
    
    print(correlationdf)       
    sns.barplot(data=correlationdf.reset_index(), x="Cell Type", y="Value", hue="Correlation", ax=ax[0])
    ax[0].set_xticks(ax[0].get_xticks())
    ax[0].set_xticklabels(labels=ax[0].get_xticklabels(), rotation=90)
    # ax[-1].set(title=f"Cmp. {cmp} V. Cell Count")
            
    