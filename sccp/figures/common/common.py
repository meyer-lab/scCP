"""
This file contains functions that are used in multiple figures.
"""
from string import ascii_letters
import sys
import time
import seaborn as sns
import pandas as pd
import matplotlib
from matplotlib import gridspec, pyplot as plt
import numpy as np
from ...crossVal import CrossVal
from ...decomposition import R2X



matplotlib.use("AGG")

matplotlib.rcParams["legend.labelspacing"] = 0.2
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["xtick.major.pad"] = 1.0
matplotlib.rcParams["ytick.major.pad"] = 1.0
matplotlib.rcParams["xtick.minor.pad"] = 0.9
matplotlib.rcParams["ytick.minor.pad"] = 0.9
matplotlib.rcParams["legend.handletextpad"] = 0.5
matplotlib.rcParams["legend.handlelength"] = 0.5
matplotlib.rcParams["legend.framealpha"] = 0.5
matplotlib.rcParams["legend.markerscale"] = 0.7
matplotlib.rcParams["legend.borderpad"] = 0.35
matplotlib.rcParams["svg.fonttype"] = "none"


def getSetup(figsize, gridd, multz=None, empts=None, constrained_layout=True):
    """Establish figure set-up with subplots."""
    sns.set(
        style="whitegrid",
        font_scale=0.7,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )

    # create empty list if empts isn't specified
    if empts is None:
        empts = []

    if multz is None:
        multz = {}

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=constrained_layout)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    x = 0
    ax = []
    while x < gridd[0] * gridd[1]:
        if x not in empts and x not in multz.keys():  # If this is just a normal subplot
            ax.append(f.add_subplot(gs1[x]))
        elif x in multz.keys():  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs1[x : x + multz[x] + 1]))
            x += multz[x]
        x += 1

    return (ax, f)


def subplotLabel(axs):
    """Place subplot labels on figure."""
    for ii, ax in enumerate(axs):
        ax.text(
            -0.2,
            1.2,
            ascii_letters[ii],
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )


def genFigure():
    """Main figure generation function."""
    fdir = "./output/"
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    exec("from sccp.figures." + nameOut + " import makeFigure", globals())
    ff = makeFigure()
    ff.savefig(fdir + nameOut + ".svg", dpi=300, bbox_inches="tight", pad_inches=0)
    ff.savefig(fdir + nameOut + ".png", dpi=300, bbox_inches="tight", pad_inches=0)

    print(f"Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n")




def flattenData(data):
    """Flattens tensor into dataframe"""
    cellCount = []
    for i in range(len(data.X_list)):
        cellCount = np.append(cellCount, data.X_list[i].shape[0])

    condNames = []

    for i in range(len(data.X_list)):
        condNames = np.append(
            condNames, np.repeat(data.condition_labels[i], cellCount[i])
        )
    flatData = np.concatenate(data.X_list, axis=0)
    dataDF = pd.DataFrame(data=flatData, columns=data.variable_labels)
    dataDF["Condition"] = condNames

    return dataDF


def flattenWeightedProjs(data, factors, projs):
    """Flattens tensor into dataframe"""
    cellCount = []
    for i in range(len(data.X_list)):
        cellCount = np.append(cellCount, data.X_list[i].shape[0])

    condNames = []

    for i in range(len(data.X_list)):
        condNames = np.append(
            condNames, np.repeat(data.condition_labels[i], cellCount[i])
        )

    weightedProjs = projs @ factors[1]

    weightedProjs = weightedProjs / np.max(np.abs(weightedProjs),axis=0)

    cmpNames = [f"Cmp. {i}" for i in np.arange(1, weightedProjs.shape[1] + 1)]
    dataDF = pd.DataFrame(data=weightedProjs, columns=cmpNames)
    dataDF["Condition"] = condNames

    return dataDF


def plotR2X(data, rank, ax):
    """Creates R2X plot for parafac2 tensor decomposition"""
    r2xError = R2X(data, rank)

    rank_vec = np.arange(1, rank + 1)
    labelNames = ["Fit: Pf2", "Fit: PCA"]
    colorDecomp = ["r", "b"]
    markerShape = ["|", "_"]

    for i in range(2):
        ax.scatter(
            rank_vec,
            r2xError[i],
            label=labelNames[i],
            marker=markerShape[i],
            c=colorDecomp[i],
            s=30.0,
        )

    ax.set(
        ylabel="Variance Explained",
        xlabel="Number of Components",
        xticks=np.linspace(0, rank, num=8, dtype=int),
        yticks=np.linspace(
            0, np.max(np.append(r2xError[0], r2xError[1])) + 0.01, num=5
        ),
    )

    ax.legend()


def plotCV(data, rank, trainPerc, ax):
    """Creates variance explained plot for parafac2 tensor decomposition CV"""
    cvError = CrossVal(data, rank, trainPerc=trainPerc)

    rank_vec = np.arange(1, rank + 1)
    labelNames = ["CV: Pf2", "CV: PCA"]
    colorDecomp = ["r", "b"]
    markerShape = ["o", "o"]

    for i in range(2):
        ax.scatter(
            rank_vec,
            cvError[i],
            label=labelNames[i],
            marker=markerShape[i],
            c=colorDecomp[i],
            s=30.0,
        )

    ax.set(
        ylabel="Variance Explained",
        xlabel="Number of Components",
        xticks=np.linspace(0, rank, num=8, dtype=int),
        yticks=np.linspace(0, np.max(np.append(cvError[0], cvError[1])) + 0.01, num=5),
    )

    ax.legend()


def plotCellTypePerExpCount(dataDF, condition, ax):
    """Plots historgram of cell counts per experiment"""
    sns.histplot(data=dataDF, x="Cell Type", hue="Cell Type", ax=ax)
    ax.set(title=condition)


def plotCellTypePerExpPerc(dataDF, condition, ax):
    """Plots historgram of cell types percentages per experiment"""
    df = dataDF.groupby(["Cell Type"]).size().reset_index(name="Count") 
    perc = df["Count"].values / np.sum(df["Count"].values)
    df["Count"] = perc
    
    sns.barplot(data=df, x="Cell Type", y="Count", ax=ax)
    ax.set(title=condition)

    
def plotGenePerCellType(genes, dataDF, ax):
    """Plots average gene expression across cell types for all conditions"""
    data = pd.melt(dataDF, id_vars=["Condition", "Cell Type"], value_vars=genes).rename(
            columns={"variable": "Gene", "value": "Value"})
    df = data.groupby(["Condition", "Cell Type", "Gene"]).mean()
    df = df.rename(columns={"Value": "Average Gene Expression For Drugs"})
    sns.stripplot(data=df, x="Gene", y="Average Gene Expression For Drugs", hue="Cell Type", dodge=True, jitter=False, ax=ax)
    


