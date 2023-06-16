"""Gene ontology for gene factors of Pf2"""
from .common import (
    subplotLabel,
    getSetup,
)
from ..geneontology import geneOntology

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 10), (6, 2))

    # Add subplot labels
    subplotLabel(ax)
    
    geneOntology(cmpNumb=24, geneAmount=50, geneValue="Overexpressed", axs=ax[0:6])
    geneOntology(cmpNumb=24, geneAmount=50, geneValue="Underexpressed", axs=ax[6:12])
    
    return f