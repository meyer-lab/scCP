from .common import getSetup
from ..imports import import_thomson
from ..imports import import_thomson
import seaborn as sns
from .figureThomsonDrop import plotDifferentialExpression

def makeFigure():
    rank = 20
    data = import_thomson()
    ax, f = getSetup((20, 10 * 7), (7 * 1, 4))
    bCellGeneSet = [
        "PXK",
        "MS4A1",
        "CD19",
        "CD74",
        "CD79A",
        "CD79B",
        "BANK1",
        "PTPRC",
        "CR2",
        "VPREB3",
    ]

    plotDifferentialExpression(data, "CTRL4", "B Cells", bCellGeneSet, rank, *ax[0:4])
    plotDifferentialExpression(
        data, "CTRL4", "B Cells", bCellGeneSet, rank, *ax[4:8], percent=0.95
    )
    plotDifferentialExpression(
        data, "CTRL4", "B Cells", bCellGeneSet, rank, *ax[8:12], percent=0.9
    )
    plotDifferentialExpression(
        data, "CTRL4", "B Cells", bCellGeneSet, rank, *ax[12:16], percent=0.8
    )
    plotDifferentialExpression(
        data, "CTRL4", "B Cells", bCellGeneSet, rank, *ax[16:20], percent=0.7
    )
    plotDifferentialExpression(
        data, "CTRL4", "B Cells", bCellGeneSet, rank, *ax[20:24], percent=0.6
    )
    plotDifferentialExpression(
        data, "CTRL4", "B Cells", bCellGeneSet, rank, *ax[24:28], percent=0.5
    )

    return f
