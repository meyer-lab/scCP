"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
from .common import (
    subplotLabel,
    getSetup,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import gseapy as gp
import pandas as pd
import mygene
import seaborn as sns



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 10), (6, 1))

    # Add subplot labels
    subplotLabel(ax)
    # over_expressed = pd.read_csv('over_expressed.txt', index_col=0)
    df = pd.read_csv('TopBotGenes_Cmp25.csv').rename(columns={"Unnamed: 0":"Gene"}).set_index("Gene")
    print(df)
    # weight = 0.01
    
    # Specifies enrichment sets to run against
    gene_sets = [
        'GO_Biological_Process_2021',
        'GO_Cellular_Component_2021',
        'GO_Molecular_Function_2021'
    ]
    
    axnumb = 0
    # Runs for each component independently
    
    num = 25
    cmpNames = ["Cmp. " + str(num)]
    
    sort_idx = np.argsort(df.to_numpy(), axis=0)
    
    
    geneAmount = 20
    genesTop = np.empty((geneAmount), dtype="<U10")
    genesBottom = np.empty((geneAmount), dtype="<U10")
    
    for i, cmp in enumerate(cmpNames):
        
        
        geness = df.index.values[sort_idx[:, num-1]]
        genesTop[:] = np.flip(geness[-geneAmount:])  
        genesBottom[:] = geness[:geneAmount]
        
        print(genesTop)
        
        # Selects genes with alpha < 0.05 for over- and under-expression
        # print(column)
        
        # print(df[cmp])
        
        # overgenes = df[cmp].loc[df[cmp] > weight]
        
        # print(overgenes)
        
        # a
        
        # over = over_expressed.loc[:, column] > weight
    
        # print(over)

        # print(over)
        # over_ensembl = list(over.index)

        # print(over_ensembl)
        # Converts ensembl to standard gene names
        # over_genes = lookup_genes(over_ensembl)
        # print(over_genes)
        # under_genes = lookup_genes(under_ensembl)

#         # Runs gene names through enrichment analyses
#         if len(over_genes) == 0:
#             over_result = None
#         else:
        over_result = gp.enrichr(
                list(overgenes.index.values),
                gene_sets=gene_sets,
                organism='Human'
            ).results
        over_result = over_result.set_index('Term', drop=True)

        print(over_result)
        
#         over_ensembl = list(over.index)

# #         # Converts ensembl to standard gene names
# #         # You may not need this if your genes are already in recognizable names!
#         over_genes = lookup_genes(over_ensembl)

#         # Runs gene names through enrichment analyses
#         if len(over_genes) == 0:
#             over_result = None
#         else:
#             over_result = gp.enrichr(
#                 list(over_genes),
#                 gene_sets=gene_sets,
#                 organism='Human'
#             ).results
#             over_result = over_result.set_index('Term', drop=True)
        
#         # print(over_result)
        
        
        # for j, gene_set in enumerate(gene_sets):
        
        #     combined = over_result.loc[
        #             over_result['Gene_set'] == gene_set,
        #             'Combined Score'
        #         ]
        #     combined = combined.sort_values(ascending=False)
        #     combined = combined.iloc[:10]
    
        #     # print(combined.index.values)
        #     combDF = pd.DataFrame({"Term": combined.index.values, "Combined Score": combined.values})
            
        #     print(combDF)

        #     p_val = over_result.loc[
        #         over_result['Gene_set'] == gene_set,
        #         'Adjusted P-value'
        #     ]
        #     p_val = p_val.sort_values(ascending=True)
        #     p_val = p_val.iloc[:10]
        #     pvalDF = pd.DataFrame({"Term": p_val.index.values, "Adjusted P-value": p_val.values})
            
        #     print(pvalDF)
        #     sns.barplot(
        #     data=combDF,
        #     x="Combined Score",
        #     y="Term",
        #     ax=ax[axnumb])
        #     pvalPlot = sns.barplot(
        #     data=pvalDF,
        #     x="Adjusted P-value",
        #     y="Term",
        #     ax=ax[axnumb+1])
        
        #     pvalPlot.set_xscale("log")
            
        #     axnumb += 2
            
        break
            
            
    
    return f

# from argparse import Namespace
# import os
# from os.path import abspath, dirname
# import sys

# import gseapy as gp
# import matplotlib.pyplot as plt
# import matplotlib.transforms as transforms
# import mygene
# from natsort import natsorted
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from sklearn.preprocessing import scale

# COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']
# PATH_HERE = dirname(abspath(__file__))


def lookup_genes(ensembl_genes, scopes='ensembl.gene'):
    """Converts ensembl gene IDs to gene names.

    Args:
        ensembl_genes (list[str]): ensembl gene IDs.
        scopes (str): mygene scope to use.

    Returns:
        symbols (list[str]): Translated gene names.
    """
    symbols = []
    ensembl_genes = [gene.split('.')[0] for gene in ensembl_genes]

    mg = mygene.MyGeneInfo()
    queries = mg.querymany(
        ensembl_genes,
        return_all=True,
        scopes=scopes
    )

    for query in queries:
        symbol = query.get('symbol')
        if symbol is not None:
            symbols.append(symbol)

    return symbols


# def fix_label(label):
#     """Splits label over two lines.

#     Args:
#         label (str): Phrase to be split.

#     Returns:
#         label (str): Provided label split over two lines.
#     """
#     if len(label) > 120:
#         label = label[:120] + '...'

#     half = int(len(label) / 2)

#     first = label[:half]
#     last = label[half:]
#     last = last.replace(' ', '\n', 1)

#     return first + last


# def main():
#     # Sets directory to save results
#     base_dir = 'output/nopv_almanac'

#     # Reads over- and under-expressed genes
    # over_expressed = pd.read_csv('data/over_expressed.txt', index_col=0)
    # under_expressed = pd.read_csv('data/under_expressed.txt', index_col=0)

#     # Specifies enrichment sets to run against
#     gene_sets = [
#         'GO_Biological_Process_2021',
#         'GO_Cellular_Component_2021',
#         'GO_Molecular_Function_2021'
#     ]

#     # Runs for each component independently
#     for column in over_expressed.columns:
#         # Configures subdirectory for component
#         directory = f'{base_dir}/component_{column}/'
#         os.makedirs(directory, exist_ok=True)

#         # Selects genes with alpha < 0.05 for over- and under-expression
#         over = over_expressed.loc[
#             over_expressed.loc[:, column] < 0.05,
#             column
#         ]
#         under = under_expressed.loc[
#             under_expressed.loc[:, column] < 0.05,
#             column
#         ]

#         over_ensembl = list(over.index)
#         under_ensembl = list(under.index)

#         # Converts ensembl to standard gene names
#         # You may not need this if your genes are already in recognizable names!
#         over_genes = lookup_genes(over_ensembl)
#         under_genes = lookup_genes(under_ensembl)

#         # Runs gene names through enrichment analyses
#         if len(over_genes) == 0:
#             over_result = None
#         else:
#             over_result = gp.enrichr(
#                 list(over_genes),
#                 gene_sets=gene_sets,
#                 organism='Human'
#             ).results
#             over_result = over_result.set_index('Term', drop=True)

#         if len(under_genes) == 0:
#             under_result = None
#         else:
#             under_result = gp.enrichr(
#                 list(under_genes),
#                 gene_sets=gene_sets,
#                 organism='Human'
#             ).results
#             under_result = under_result.set_index('Term', drop=True)

#         # Plots out results
#         # Produces four plots depicting (clockwise from top-left):
#         # Combined Score for over-expressed, combined score for under-expressed,
#         # fisher's exact test for over-expressed, fisher's exact test for
#         # under-expressed
#         for gene_set in gene_sets:
#             fig, axs = plt.subplots(
#                 2,
#                 2,
#                 figsize=(8, 8),
#                 constrained_layout=True
#             )
#             axs[1, 1].set_xlabel('Fisher Exact Test P-value')
#             axs[1, 0].set_xlabel('Combined Score')
#             axs[0, 0].set_ylabel('Overexpressed Genes')
#             axs[1, 0].set_ylabel('Underexpressed Genes')

#             for i in range(2):
#                 for j in range(2):
#                     axs[i, j].set_yticks([])

#             for result, row in zip([over_result, under_result], [0, 1]):
#                 if result is None:
#                     continue

#                 combined = result.loc[
#                     result['Gene_set'] == gene_set,
#                     'Combined Score'
#                 ]
#                 combined = combined.sort_values(ascending=False)
#                 combined = combined.iloc[:10]

#                 p_val = result.loc[
#                     result['Gene_set'] == gene_set,
#                     'Adjusted P-value'
#                 ]
#                 p_val = p_val.sort_values(ascending=True)
#                 p_val = p_val.iloc[:10]

#                 axs[row, 0].barh(
#                     range(len(combined)),
#                     combined,
#                     color=COLOR_CYCLE[1]
#                 )
#                 axs[row, 1].plot(
#                     [0.05, 0.05],
#                     [-1, 100],
#                     alpha=0.25,
#                     color='k',
#                     linestyle='--'
#                 )
#                 axs[row, 1].barh(
#                     range(len(p_val)),
#                     p_val,
#                     color=COLOR_CYCLE[1]
#                 )

#                 axs[row, 1].set_xscale('log')
#                 axs[row, 1].set_xlim(right=10)
#                 axs[row, 1].set_ylim([-1, 10])
#                 axs[row, 0].set_ylim([-1, 10])

#                 c_transform = transforms.blended_transform_factory(
#                     axs[row, 0].transAxes,
#                     axs[row, 0].transData
#                 )
#                 p_transform = transforms.blended_transform_factory(
#                     axs[row, 1].transAxes,
#                     axs[row, 1].transData
#                 )

#                 for i in range(len(combined)):
#                     combined_label = combined.index[i]
#                     p_label = p_val.index[i]
#                     if len(combined_label) > 60:
#                         combined_label = fix_label(combined_label)
#                     if len(p_label) > 60:
#                         p_label = fix_label(p_label)

#                     axs[row, 0].text(
#                         0.025,
#                         i,
#                         combined_label,
#                         fontsize=7,
#                         va='center',
#                         ha='left',
#                         transform=c_transform
#                     )
#                     axs[row, 1].text(
#                         0.025,
#                         i,
#                         p_label,
#                         fontsize=7,
#                         va='center',
#                         ha='left',
#                         transform=p_transform
#                     )

#             fig.suptitle(gene_set.replace('_', ' ') + f' - Component {column}')
#             fig.savefig(f'{directory}/{gene_set}.png')


# if __name__ == '__main__':
#     main()
