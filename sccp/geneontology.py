import numpy as np
import gseapy as gp
import pandas as pd
import seaborn as sns


def geneOntology(cmpNumb: int, geneAmount, geneValue, axs):
        """Plots top Gene Ontology terms for molecular function, 
        biological process, cellular component. Uses factors as 
        input for function"""
        
        df = pd.read_csv("sccp/data/TopBotGenes_Cmp25.csv").rename(columns={"Unnamed: 0":"Gene"}).set_index("Gene")
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
        
        
        for i in range(len(geneSets)):
            if geneValue == "Overexpressed":
                enrichrTopGO = runGO(genesTop, geneSets)
                CombGO = combinedDF(enrichrTopGO, geneSets[i])
                PvalGO = pvalueDF(enrichrTopGO, geneSets[i])
            else:
            
                enrichrBotGO = runGO(genesBottom, geneSets)
                CombGO = combinedDF(enrichrBotGO, geneSets[i])
                PvalGO = pvalueDF(enrichrBotGO, geneSets[i])
            
            plotCombGO(CombGO, geneValue, ax=axs[2*i])
            plotPvalGO(PvalGO, geneValue, ax=axs[(2*i)+1])
            
    
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
    
def combinedDF(enrichrGO, geneSet):
    """Saves combined score for gene terms in DF"""
     # Combined Score
    combined = enrichrGO.loc[
            enrichrGO["Gene_set"] == geneSet,
            "Combined Score"
            ]
    combined = combined.sort_values(ascending=False)
    combined = combined.iloc[:10]
    combDF = pd.DataFrame({"Term": combined.index.values, "Combined Score": combined.values})
    
    return combDF

def pvalueDF(enrichrGO, geneSet):
    """Saves adjusted p value for gene terms in DF"""
    p_val = enrichrGO.loc[
        enrichrGO["Gene_set"] == geneSet,
        "Adjusted P-value"
            ]
    p_val = p_val.sort_values(ascending=True)
    p_val = p_val.iloc[:10]
    pvalDF = pd.DataFrame({"Term": p_val.index.values, "Adjusted P-value": p_val.values})
    
    return pvalDF

def plotCombGO(GO, geneValue, ax):
    """Plots combines score for gene ontology"""
    sns.barplot(
    data=GO,
    x="Combined Score",
    y="Term",
    ax=ax)
    ax.set_title(geneValue + "-Genes")
        
def plotPvalGO(GO, geneValue, ax):
    """Plots adjusted p value for gene ontology"""
    pvalPlot = sns.barplot(
    data=GO,
    x="Adjusted P-value",
    y="Term",
    ax=ax)
    ax.set_title(geneValue + "-Genes")
    pvalPlot.set_xscale("log")
