"""
Lupus: Plots of amount of cells and cell type distribution across all experiments
"""
from anndata import read_h5ad
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import population_bar_chart
from .figureLupus17 import getCellCountPercDF
import statsmodels.api as sm


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 6), (4, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")
    
    celltype = ["Cell Type", "Cell Type2", "leiden"]
    cellperc = [True, False]
    label = ["Cell Type Percentage", "Cell Count"]
    plot = 0

    for i in range(len(celltype)):
        for j in range(len(cellperc)):
            # cellPercDF = getCellCountPercDF(X, celltype=celltype[i], cellPerc=cellperc[j])
            # getCellCountPercDF(X, celltype=celltype[i], cellPerc=cellperc[j])
            df = getcellsDF(X, celltype=celltype[i])
            pdf = diff_abund_test(df)
            print(pdf)
            
            # sns.boxplot(data=cellPercDF, x="Cell Type", y=label[j], 
            #             hue="Status", showfliers=False, ax=ax[plot])
            # rotate_axis(ax[plot])
            # plot+=1
       
    plotOverallCellCount(X, ax[6])
    f.delaxes(ax[7])   
            


    return f
    
def plotOverallCellCount(X, ax):
    df = X.obs[["SLE_status", "Condition"]].reset_index(drop=True)
    dfCond = df.groupby(["Condition","SLE_status"], observed=True).size().reset_index(name="Cell Count")
    
    sns.boxplot(data=dfCond, x="SLE_status", y="Cell Count", hue="SLE_status", showfliers=False, ax=ax)
    
    return 

def rotate_axis(ax):
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)
    
    

def getcellsDF(X, celltype="Cell Type", cellPerc=True):
    """Determine cell count or cell type percentage per condition and patient"""
    df = X.obs[["Cell Type", "SLE_status", "Condition", "Cell Type2", "leiden"]].reset_index(drop=True)
    dfCond = (
        df.groupby(["Condition"], observed=True).size().reset_index(name="Cell Number")
    )
    dfCellType = (
        df.groupby([celltype, "Condition", "SLE_status"], observed=True)
        .size()
        .reset_index(name="Count")
    )
    dfCellType["Count"] = dfCellType["Count"].astype("float")
    
    if cellPerc is True:
        for i, cond in enumerate(np.unique(df["Condition"])):
            dfCellType.loc[dfCellType["Condition"] == cond, "Cell Type Percentage"] = (
                100
                * dfCellType.loc[dfCellType["Condition"] == cond, "Count"].to_numpy()
                / dfCond.loc[dfCond["Condition"] == cond]["Cell Number"].to_numpy()
            )
    
    dfCellType.rename(columns={celltype: "Cell Type", "SLE_status": "Status"}, inplace=True)
    
    return dfCellType



def diff_abund_test(cellDF):
    """Calculates whether cells are statistically signicantly different"""
    pvalDF = pd.DataFrame()
    cellDF["Y"] = 1
    cellDF.loc[cellDF["Status"]== "SLE", "Y"] = 0
    for cell in cellDF["Cell Type"].unique():
        Y = cellDF.loc[cellDF["Cell Type"] == cell]["Cell Type Percentage"].values
        X = cellDF.loc[cellDF["Cell Type"] == cell].Y.values
        weights = np.power(cellDF.loc[cellDF["Cell Type"] == cell]["Count"].values, 1)
        mod_wls = sm.WLS(Y, sm.tools.tools.add_constant(X), weights=weights)
        res_wls = mod_wls.fit()
        pvalDF = pd.concat(
            [
                pvalDF,
                pd.DataFrame(
                    {
                        "Cell Type": [cell],
                        "p Value": res_wls.pvalues[1]
                        * cellDF["Cell Type"].unique().size,
                    }
                ),
            ]
        )

    return pvalDF
