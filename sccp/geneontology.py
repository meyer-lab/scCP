import pandas as pd
import gseapy as gp
from anndata import AnnData


def geneOntology(X: AnnData, cmpNumb: int) -> pd.DataFrame:
    """Plots top Gene Ontology terms for molecular function,
    biological process, cellular component. Uses factors as
    input for function"""

    # Specifies enrichment sets to run against
    geneSets =   [
        "GO_Molecular_Function_2021",
    ]
    # geneSets =   [
    #     "GO_Cellular_Component_2021",
    # ]
    # geneSets = ["MSigDB_Hallmark_2020" ]
    geneSets = ["KEGG_2021_Human"]
    # geneSets = [
    #     "GO_Biological_Process_2021",
    #     "GO_Cellular_Component_2021",
    #     "GO_Molecular_Function_2021",
    # ]


    geneRank = pd.Series(X.varm["Pf2_C"][:, cmpNumb - 1], index=X.var_names)
    geneRank = geneRank.sort_values()
    
    geneRank = geneRank.index.values[-50:]
    
    
    for i in geneRank:
        print(i)
    # a
    
    print( geneRank.iloc[-50:])



    df = gp.prerank(
        geneRank,
        gene_sets=geneSets,
        organism="Human",
        no_plot=True,
        verbose=True,
        threads=30,
    ).res2d
    
    
    
    # df = gp.enrichr(gene_list=geneRank.index.tolist(),
    #              gene_sets=geneSets,
    #             organism="Human",
    #              outdir=None)
    
    # df = df.loc[df["FWER p-val"] < 0.05]

    return df
