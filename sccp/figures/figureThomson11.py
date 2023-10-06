"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
"""
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
    flattenWeightedProjs,
)
from ..imports.scRNA import ThompsonXA_SCGenes, tensorFy
from ..imports.gating import gateThomsonCells
import seaborn as sns


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes()
    pfdata = tensorFy(data, "Drugs")

    _, factors, projs = openPf2(rank=30, dataName="Thomson")

    weightedProjDF = flattenWeightedProjs(pfdata, factors[1], projs)
    weightedProjDF["Cell Type"] = gateThomsonCells()
    weightedProjDF.sort_values(by=["Condition", "Cell Type"], inplace=True)

    genes = [
        "VPREB3",
        "CD79A",
        "FAM111B",
        "HOPX",
        "SLC30A3",
        "MS4A1",
        "CCNB1",
        "GADD45A",
        "SLC40A1",
        "CDC20",
        "ITIH3",
        "CPEB1",
    ]
    cmp = 20

    for i, gene in enumerate(genes):
        df = weightedProjDF[["Cell Type", "Cmp. " + str(cmp)]]
        df.loc[:, gene] = data[:, gene].X

        sns.scatterplot(
            data=df, x="Cmp. " + str(cmp), y=gene, hue="Cell Type", ax=ax[i]
        )

    return f
