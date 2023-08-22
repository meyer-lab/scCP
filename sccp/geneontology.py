import numpy as np
import gseapy as gp
import pandas as pd
import seaborn as sns
#from .figures.common import plotCombGO, plotPvalGO

def geneOntology(cmpNumb: int, geneAmount, goTerms, geneValue):
    """Plots top Gene Ontology terms for molecular function, 
    biological process, cellular component. Uses factors as 
    input for function"""
    
    df = pd.read_csv("sccp/data/Thomson/TopBotGenes_Cmp30.csv").rename(columns={"Unnamed: 0":"Gene"}).set_index("Gene")
    sort_idx = np.argsort(df.to_numpy(), axis=0)
    
    # Specifies enrichment sets to run against
    geneSets = [
        "GO_Biological_Process_2021",
        "GO_Cellular_Component_2021",
        "GO_Molecular_Function_2021"]

    genesTop = np.empty((geneAmount), dtype="<U10")
    genesBottom = np.empty((geneAmount), dtype="<U10")
    
    geneNames = df.index.values[sort_idx[:, cmpNumb-1]]
    genesTop[:] = np.flip(geneNames[-geneAmount:])  
    genesBottom[:] = geneNames[:geneAmount]
    
    totalCombDF = pd.DataFrame()
    totalPvalDF = pd.DataFrame()
    for i, geneSet in enumerate(geneSets):
        if geneValue == "Overexpressed":
            enrichrGO = runGO(genesTop, geneSets)
        else:
            enrichrGO = runGO(genesBottom, geneSets)
            
        CombGO = combinedDF(enrichrGO, geneSet, geneValue, goTerms)
        PvalGO = pvalueDF(enrichrGO, geneSet, geneValue, goTerms)
        
        totalCombDF = pd.concat([totalCombDF, CombGO])
        totalPvalDF = pd.concat([totalPvalDF, PvalGO])
        
    return totalCombDF, totalPvalDF

            
    
def runGO(geneList, geneSets):
    """GSEApy is a Python/Rust implementation of GSEA and wrapper for Enrichr.
    Uses enrichment sets to run against"""
    enrichrGO = gp.enrichr(
            list(geneList),
                gene_sets=geneSets,
                organism="Human"
                ).results
    enrichrGO = enrichrGO.set_index("Term", drop=True)
    
    return enrichrGO
    
def combinedDF(enrichrGO, geneSet, geneValue, goTerms):
    """Saves combined score for gene terms in DF"""
     # Combined Score
    combined = enrichrGO.loc[
            enrichrGO["Gene_set"] == geneSet,
            "Combined Score"
            ]
    combined = combined.sort_values(ascending=False)
    combined = combined.iloc[:goTerms]
    combDF = pd.DataFrame({"Term": combined.index.values, "Combined Score": combined.values, "Expression": geneValue, "Gene Set": geneSet})
    
    return combDF

def pvalueDF(enrichrGO, geneSet, geneValue, goTerms):
    """Saves adjusted p value for gene terms in DF"""
    p_val = enrichrGO.loc[
        enrichrGO["Gene_set"] == geneSet,
        "Adjusted P-value"
            ]
    p_val = p_val.sort_values(ascending=True)
    p_val = p_val.iloc[:goTerms]
    pvalDF = pd.DataFrame({"Term": p_val.index.values, "Adjusted P-value": p_val.values, "Expression": geneValue, "Gene Set": geneSet})
    
    return pvalDF

def addGOCol(seriez, data, geneset, term: int = 1, verbose = False):
    """Adds/Updates a column to/from the input dataset called 'GO_term' that contains a GO term
    relevant to that gene. To do this, it finds the top 10 enriched go terms in the set it was given,
    defaulting to using GO_Biological_Process_2018. If `verbose = True`, it will print out these terms
    as it runs, and subsequent runs can select GO terms to color manually using `term`.

    Requires an input `seriez`, which is a pandas series of genes in data. Used with common.plotTopGenes
    """
    term = term - 1
    enrichrGO = runGO(seriez.index, geneset)
    # give info about which ones they can choose
    if verbose == True:
        print('Top 10 GO terms:\n')
        for i in range(10):
            print(i + 1, enrichrGO.index[i])
        print('\nYou Selected:\n', enrichrGO.index[term])

    genes_from_term = enrichrGO['Genes'][term].split(";")
    term = enrichrGO.index[term]
    if 'GO_term' not in data.columns:
        data['GO_term'] = np.where(data['Gene ID'].isin(genes_from_term), term, 'neither')
    else:
        data['GO_term'] = np.where(data['Gene ID'].isin(genes_from_term), term, data['GO_term'])

    
    return data
