"""
Lupus: Gene ontology for gene factors of Pf2
"""
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
from ..geneontology import geneOntology
from gseapy import barplot, dotplot
import pandas as pd

import seaborn as sns
from .commonFuncs.plotFactors import bot_top_genes



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 6), (2, 3))

    # Add subplot labels
    subplotLabel(ax)


    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")
    
    # amount = 50
    # genes = bot_top_genes(X, cmp=4, geneAmount=amount)
    
    # topgenes = genes[amount:]
    # botgenes = genes[:amount]
    
    # for i in topgenes:
    #     print(i) 
    
    

    
    
    
    df = pd.read_csv("botgene_cmp4.csv", dtype=str)

    # df = df.drop(columns=["ID", "Verbose ID", "Unnamed: 55", 
    #                 "Unnamed: 56", "Unnamed: 57",
    #                 "Unnamed: 58", "Unnamed: 59"])
    df = df.drop(columns=["ID", "Verbose ID"])


    
    category= df["Category"].to_numpy().astype(str)
    
    df = df.drop(columns=["Category"])
    df["Process"] = category
    df = df.iloc[:1000, :]
    print(df)
    df["Total Genes"] = df.iloc[:, 2:-1].astype(int).sum(axis=1).to_numpy()


    # df = df.loc(df["Process"] == "GO: Biological Process")

    df= df.loc[df.loc[:, "Process"] == "GO: Biological Process"]
    df["pValue"] = df["pValue"].astype(float)
    
    sns.scatterplot(data=df.iloc[:10, :], x="pValue", y="Name", hue="Total Genes", ax=ax[0])
    ax[0].set(xscale="log")
    
    
    














    # df = geneOntology(X, 14)




    # # a = dotplot(df, column="FDR q-val", ax=ax[0], ofname=ax[0])
    # a = barplot(df, x="Gene_set", column="FDR q-val", ax=ax[0], ofname="save.png")
    # # a = dotplot(df.res2d, ax=ax[0], ofname="save.png")
    # # a.imshow()
    # f.show()
    










    # geneSets = [
    #     "GO_Biological_Process_2021",
    #     "GO_Cellular_Component_2021",
    #     "GO_Molecular_Function_2021",
    # ]







    return f
