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

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 10), (2, 3))

    # Add subplot labels
    subplotLabel(ax)


    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")













    df = geneOntology(X, 27)




    # a = dotplot(df, column="FDR q-val", ax=ax[0], ofname=ax[0])
    a = barplot(df, x="Gene_set", column="FDR q-val", ax=ax[0], ofname="save.png")
    # a = dotplot(df.res2d, ax=ax[0], ofname="save.png")
    # a.imshow()
    f.show()
    










    # geneSets = [
    #     "GO_Biological_Process_2021",
    #     "GO_Cellular_Component_2021",
    #     "GO_Molecular_Function_2021",
    # ]







    return f
