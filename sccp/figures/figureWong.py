"""
Figure 5a_e Generation Script

This module generates a comprehensive figure comparing PCA and PF2 component analyses 
for gene loadings and factors.
"""

import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import mygene
from .common import getSetup, subplotLabel
import statsmodels.api as sm


def makeFigure():
    ax, f = getSetup((22, 9), (4, 10))
    subplotLabel(ax)
    
    geneAmount = 10
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    pc1_load = load_pc_loadings(pc_component=1)
    pc2_load = load_pc_loadings(pc_component=2)
    
    overlap_genes = list(set(pc1_load['Gene']) & set(X.var_names))
    
    pcs = ["PC1", "PC2"]
    pc_loads = [pc1_load, pc2_load]
    axs = 0
    for i, pc in enumerate(pcs):
        pc_load = pc_loads[i][pc_loads[i]['Gene'].isin(overlap_genes)].sort_values(by=pc, ascending=False)
        genes = pc_load['Gene'].values
        for j, gene in enumerate(np.concatenate([genes[:geneAmount], genes[-geneAmount:]])):
            df = avegene_lupus(X, gene) 
            pval_df = wls_stats_comparison(
                df,
                column_comparison_name=gene,
                category_name="Status",
                status_name="SLE",
            )
            sns.boxplot(data=df, x="Cell Type", y=gene, hue="Status", 
                        showfliers=False, ax=ax[axs])
            
            
            current_ylim = ax[axs].get_ylim()
            ax[axs].set_ylim(current_ylim[0], current_ylim[1] * 1.2)
            add_pvalue_annotation(ax[axs], pval_df)
            
            
            ax[axs].set_ylabel(f"{gene} Expression")
            ax[axs].set_ylim(bottom=0)            
            ax[axs].locator_params(axis="y", nbins=4)
            
            if 0 <= axs < 10:
                ax[axs].set_title("PC1 Pos")
            elif 10 <= axs < 20:
                ax[axs].set_title("PC1 Neg")
            elif 20 <= axs < 30:
                ax[axs].set_title("PC2 Pos")
            elif axs >= 30:
                ax[axs].set_title("PC2 Neg")
                
            axs += 1
            

    return f



def load_pc_loadings(pc_component: int) -> pd.DataFrame:
    """
    Load and preprocess PC loadings for a specific component.

    Args:
        pc_component (int): Principal Component number (1 or 2)
        geneAmount (int): Number of genes to consider

    Returns:
        pd.DataFrame: Processed PC loadings DataFrame
    """

    df = pd.read_csv(f"loadings_time_series_PC{pc_component}.csv", dtype=str)
    df = df.rename(columns={"Unnamed: 0": "Gene"})
    df["Gene"] = convert_gene_symbols(df["Gene"])[0]
    df[f"PC{pc_component}"] = stats.zscore(df[f"PC{pc_component}"].astype(float))

    return df

def load_pf2_loadings(X) -> pd.DataFrame:
    """
    Load and preprocess PC loadings for a specific component.

    Args:
        pc_component (int): Principal Component number (1 or 2)
        geneAmount (int): Number of genes to consider

    Returns:
        pd.DataFrame: Processed PC loadings DataFrame
    """

    df = pd.DataFrame(data=X.varm['Pf2_C'], columns=[f"Pf2_{i}" for i in range(1, X.varm['Pf2_C'].shape[1]+1)])
    df =  df.set_index(X.var_names).reset_index().rename(columns={"index": "Gene"})

    return df
    


def convert_gene_symbols(genes):
    """
    Convert gene symbols using MyGene.

    Args:
        genes (list): List of gene symbols
        species (str): Species for gene conversion

    Returns:
        tuple: Converted genes and list of genes without hits
    """
    mg = mygene.MyGeneInfo()
    results = mg.querymany(
        genes, 
        scopes='symbol', 
        fields=['symbol', 'entrezgene'], 
        species='mouse', 
        transformed=True
    )

    conversion_map = []
    no_hit_genes = []
    
    for gene, result in zip(genes, results):
        if result and 'symbol' in result:
            conversion_map.append(result.get('symbol', gene).upper())
        else:
            conversion_map.append(gene.upper())
            no_hit_genes.append(gene)
    
    return conversion_map, no_hit_genes


def avegene_lupus(X, gene):
    genesV = X[:, gene]
    df = genesV.to_df()
    
    df["Status"] = X.obs["SLE_status"].values
    df["Condition"] = X.obs["Condition"].values
    df["Cell Type"] = X.obs["Cell Type"].values
    
    df = df[df["Cell Type"].isin(["nCM", "CM"])]
    df_mean = df.groupby(["Status", "Cell Type", "Condition"], observed=False).mean().reset_index(
        ).dropna().sort_values(["Cell Type", "Condition"])
    
    df_count = df.groupby(["Cell Type", "Condition"], observed=False).size().reset_index(
        name="Cell Count").sort_values(["Cell Type", "Condition"])
    df_count = df_count[df_count["Cell Type"].isin(["nCM", "CM"])]

    df_mean["Cell Count"] = df_count["Cell Count"].to_numpy()
    df_mean['Cell Type'] = df_mean['Cell Type'].cat.remove_unused_categories()
    
    return df_mean


def wls_stats_comparison(df, column_comparison_name, category_name, status_name):
    """Calculates whether cells are statistically signicantly different"""
    pval_df = pd.DataFrame()
    df["Y"] = 1
    df.loc[df[category_name] == status_name, "Y"] = 0
    for cell in df["Cell Type"].unique():
        Y = df.loc[df["Cell Type"] == cell][column_comparison_name].to_numpy()
        X = df.loc[df["Cell Type"] == cell]["Y"].to_numpy()
        weights = np.power(df.loc[df["Cell Type"] == cell]["Cell Count"].values, 1)
        mod_wls = sm.WLS(Y, sm.tools.tools.add_constant(X), weights=weights)
        res_wls = mod_wls.fit()
        pval_df = pd.concat(
            [
                pval_df,
                pd.DataFrame(
                    {
                        "Cell Type": [cell],
                        "p Value": res_wls.pvalues[1]
                        * df["Cell Type"].unique().size,
                    }
                ),
            ]
        )

    return pval_df


def add_pvalue_annotation(ax, pval_df):
    # Get the current y-axis limit
    y_max = ax.get_ylim()[1]
    
    # Determine where to place the p-value bars
    spacing = y_max * 0.1
    
    # Iterate through unique cell types
    for i, cell_type in enumerate(pval_df.index):
        p_value = pval_df.loc[cell_type, 'p Value']
        
 
        # Determine significance stars
        if p_value.iloc[i] < 0.001:
            stars = '***'
        elif p_value.iloc[i] < 0.01:
            stars = '**'
        elif p_value.iloc[i] < 0.05:
            stars = '*'
        else:
            stars = 'ns'
        
        # Position for annotation
        y_position = y_max + spacing * (i + 1)
        
        # Add p-value text
        ax.text(i, y_position, f'p = {p_value.iloc[i]:.3f} {stars}', 
                horizontalalignment='center', 
                verticalalignment='bottom')
        
        # Optional: Add a horizontal line to indicate p-value comparison
        ax.plot([i-0.3, i+0.3], [y_position]*2, color='black', linewidth=1)