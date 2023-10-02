from sccp.figures.common import (
    subplotLabel,
    getSetup,
    plotFactorsRelaxed,
    plotFactors,
    plotR2X,
    plotCV,
    plotWeight,
    openPf2,
    savePf2,
    plotCellTypeUMAP,
    flattenData,
    openUMAP,
)
from sccp.imports.scRNA import ThompsonXA_SCGenesAD, tensorFy
from sccp.imports.gating import gateThomsonCells
from parafac2 import parafac2_nd
import umap
import numpy as np
import json
import seaborn as sns


def makeFigure():
    ax, f = getSetup((18, 16), (2, 2))
    subplotLabel(ax)

    data = ThompsonXA_SCGenesAD()
    rank = 30

    data.obs["cell_type"] = gateThomsonCells(rank)

    cell_type = data.obs.cell_type.unique()[1]
    drug = data.obs.Drugs.unique()[0]
    print(cell_type)
    print(len(data))
    print(drug)
    data_subset = data[(data.obs.cell_type != cell_type) | (data.obs.Drugs != drug)]
    print(len(data_subset))
    dataDF = tensorFy(data_subset, "Drugs")
    weight, factors, projs, r2x = parafac2_nd(
        dataDF,
        rank=rank,
        random_state=1,
    )

    gene_factors = factors[2]
    idx = np.where(dataDF.condition_labels == drug)[0][0]
    large_comps = np.where(factors[0][idx] > 0.2)[0]
    gene_factors_T = np.transpose(gene_factors)
    k = 0
    for i in large_comps:
        ax[k].set_title(f"Component {i+1} most upregulated and downregulated genes")
        t_five_genes = np.argpartition(gene_factors_T[i], -5)[-5:]
        b_five_genes = np.argpartition(gene_factors_T[i], 5)[:5]
        gene_names = np.concatenate(
            (dataDF.variable_labels[t_five_genes], dataDF.variable_labels[b_five_genes])
        )
        gene_values = np.concatenate(
            (gene_factors_T[i][t_five_genes], gene_factors_T[i][b_five_genes])
        )
        gene_values, gene_names = np.transpose(sorted(zip(gene_values, gene_names)))
        sns.barplot(x=gene_names, y=gene_values.astype(float), ax=ax[k], color="gray")
        k += 1

    return f
