"""
Thomson dataset: Cell counts and cell type percentages per condition.
"""
from anndata import read_h5ad
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 18), (8, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad", backed="r")

    df = pd.DataFrame(
        {"Cell Type": X.obs["Cell Type2"], "Condition": X.obs["Condition"]}
    )

    # Per condition counts
    dfCond = (
        df.groupby(["Condition"], observed=True).size().reset_index(name="Cell Number")
    )
    
    # sns.barplot(data=dfCond, x="Condition", y="Cell Number", color="k", ax=ax[0])
    # ax[0].set(ylabel="Number of Cells")
    # rotate_axis(ax[0])


    # Per condition cell type percentages
    dfCellType = (
        df.groupby(["Cell Type", "Condition"], observed=True)
        .size()
        .reset_index(name="Cell Count")
    )
    # dfCellType["Cell Type Percentage"] = (
    #     100
    #     * dfCellType["Count"]
    #     / dfCellType.groupby("Condition", observed=True)["Count"].transform("sum")
    # )

    for i, (name, group) in enumerate(dfCellType.groupby("Cell Type", observed=True)):
        sns.barplot(
            data=group,
            x="Condition", 
            y="Cell Count",
            color="k",
            ax=ax[i],
        )
        rotate_axis(ax[i])
        ax[i].set(ylabel=f"{name} Count")
        
    f.delaxes(ax[13])   
    f.delaxes(ax[14]) 
    f.delaxes(ax[15])     
       


    return f

def rotate_axis(ax):
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)