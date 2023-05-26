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
    
    cmpNumb = 24
    topGO, bottomGO = genesGO(cmpNumb, geneAmount=40)
    
    print(topGO[0])

    
    
        
        
    
    return f


def genesGO(cmpNumb: int, geneAmount=35):
        """Plots top Gene Ontology terms for molecular function, 
        biological process, cellular component. Uses factors as 
        input for function"""
        
        df = pd.read_csv("TopBotGenes_Cmp25.csv").rename(columns={"Unnamed: 0":"Gene"}).set_index("Gene")
        sort_idx = np.argsort(df.to_numpy(), axis=0)
        
        # Specifies enrichment sets to run against
        geneSets = [
            "GO_Biological_Process_2021",
            "GO_Cellular_Component_2021",
            "GO_Molecular_Function_2021"]

        genesTop = np.empty((geneAmount), dtype="<U10")
        genesBottom = np.empty((geneAmount), dtype="<U10")

        topCombGO = np.empty((len(geneSets)))
        topPvalGO = np.empty((len(geneSets)))
        botCombGO = np.empty((len(geneSets)))
        botPvalGO = np.empty((len(geneSets)))
        
        geneNames = df.index.values[sort_idx[:, cmpNumb-1]]
        genesTop[:] = np.flip(geneNames[-geneAmount:])  
        genesBottom[:] = geneNames[:geneAmount]
        
        for i in range(len(geneSets)):
            enrichrTopGO = runGO(genesTop, geneSets)
            topCombGO[i] = combinedDF(enrichrTopGO, geneSets[i])
            topPvalGO[i] = pvalueDF(enrichrTopGO, geneSets[i])
            
            enrichrBotGO = runGO(genesBottom, geneSets)
            botCombGO[i] = combinedDF(enrichrBotGO, geneSets[i])
            botPvalGO[i] = pvalueDF(enrichrBotGO, geneSets[i])
        
        return  [topCombGO, topPvalGO], [botCombGO, botPvalGO]
    
    
    
def runGO(geneList, geneSets):
    enrichrGO = gp.enrichr(
            list(geneList),
                gene_sets=geneSets,
                organism='Human'
                ).results
    enrichrGO = enrichrGO.set_index('Term', drop=True)
    
    return enrichrGO
    
def combinedDF(enrichrGO, geneSet):
     # Combined Score
    combined = enrichrGO.loc[
            enrichrGO["Gene_set"] == geneSet,
            "Combined Score"
            ]
    combined = combined.sort_values(ascending=False)
    combined = combined.iloc[:10]
    combDF = pd.DataFrame({"Term": combined.index.values, "Combined Score": combined.values})
    
    return combinedDF

def pvalueDF(enrichrGO, geneSet):
     # Combined Score
    p_val = enrichrGO.loc[
        enrichrGO['Gene_set'] == geneSet,
        'Adjusted P-value'
            ]
    p_val = p_val.sort_values(ascending=True)
    p_val = p_val.iloc[:10]
    pvalDF = pd.DataFrame({"Term": p_val.index.values, "Adjusted P-value": p_val.values})
    
    return pvalDF
