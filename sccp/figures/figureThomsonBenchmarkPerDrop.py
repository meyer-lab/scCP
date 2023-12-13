from .common import getSetup
from ..imports import import_thomson
from ..factorization import pf2
from .commonFuncs.plotFactors import (
    plotConditionsFactors,
    plotCellState,
    plotGeneFactors,
)
from tlviz.factor_tools import factor_match_score as fms
from tensorly.cp_tensor import CPTensor
from ..imports import import_thomson
from .figureThomson1 import groupDrugs
import numpy as np


def makeFigure():
    rank = 20
    data = import_thomson()

    drugs_to_drop = ["CTRL4", "Alprostadil", "Betamethasone Valerate"]
    fmsScores = {}
    # sampled_data_all_bcells = data[(data.obs["Cell Type"] != "B Cells")]

    for drug_to_drop in drugs_to_drop:
        # sampled_data_tcells = data[
        #     (data.obs["Cell Type"] != "T Cells") | (data.obs["Condition"] != drug_to_drop)
        # ]

        sampled_data_bcells = data[
            (data.obs["Cell Type"] != "B Cells")
            | (data.obs["Condition"] != drug_to_drop)
        ]

        # sampled_data_nkcells = data[
        #     (data.obs["Cell Type"] != "NK Cells") | (data.obs["Condition"] != drug_to_drop)
        # ]

        # sampled_data_monocytes = data[
        #     (data.obs["Cell Type"] != "Monocytes") | (data.obs["Condition"] != drug_to_drop)
        # ]

        # sampled_data_DCs = data[
        #     (data.obs["Cell Type"] != "DCs") | (data.obs["Condition"] != drug_to_drop)
        # ]

        # sampled_data_thelpers = data[
        #     (data.obs["Cell Type"] != "T Helpers") | (data.obs["Condition"] != drug_to_drop)
        # ]

        ax, f = getSetup((20, 20), (1, 1))

        all_sampled_data = [
            data,
            sampled_data_bcells,
            # sampled_data_all_bcells
            # sampled_data_tcells,
            # sampled_data_nkcells,
            # sampled_data_monocytes,
            # sampled_data_DCs,
            # sampled_data_thelpers,
        ]
        all_sampled_data_names = [
            "All",
            "B Cells",
            # "All B Cells"
            # "T Cells",
            # "NK Cells",
            # "Monocytes",
            # "DCs",
            # "T Helpers",
        ]

        all_X = []

        for sampled_data, name, i in zip(
            all_sampled_data, all_sampled_data_names, np.arange(7)
        ):
            print(f"Running {name}")
            sampledX = pf2(sampled_data, rank, doEmbedding=False)
            all_X.append(sampledX)

        factors = [all_X[0].uns["Pf2_A"], all_X[0].uns["Pf2_B"], all_X[0].varm["Pf2_C"]]
        dataXcp = CPTensor(
            (
                all_X[0].uns["Pf2_weights"],
                factors,
            )
        )

        for i in range(1, 2):
            sampled_factors = [
                all_X[i].uns["Pf2_A"],
                all_X[i].uns["Pf2_B"],
                all_X[i].varm["Pf2_C"],
            ]
            sampledXcp = CPTensor(
                (
                    all_X[i].uns["Pf2_weights"],
                    sampled_factors,
                )
            )
            fmsScore = fms(
                dataXcp,
                sampledXcp,
                consider_weights=True,
                skip_mode=None,
            )

            fmsScores[drug_to_drop] = fmsScore

            print(f"Factor score match for {all_sampled_data_names[i]}: {fmsScore}")

        # sampled_factors = [
        #         all_X[2].uns["Pf2_A"],
        #         all_X[2].uns["Pf2_B"],
        #         all_X[2].varm["Pf2_C"],
        #     ]
        # sampledXcp = CPTensor(
        #     (
        #         all_X[2].uns["Pf2_weights"],
        #         sampled_factors,
        #     )
        # )
        # fmsScore = fms(
        #     dataXcp,
        #     sampledXcp,
        #     consider_weights=True,
        #     skip_mode=None,
        # )
        # fmsScores["All Conditions"] = fmsScore

    
    ax[0].bar(*zip(*fmsScores.items()))
    ax[0].set_title(
        "Factor Score Match for Thomson Dataset when dropping different conditions for b cells"
    )

    return f
