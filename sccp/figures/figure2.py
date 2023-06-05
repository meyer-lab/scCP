"""
Gene ontology for gene factors of Pf2
"""
from .common import (
    subplotLabel,
    getSetup,
)
from ..geneontology import geneOntology

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 6), (2, 1))

    # Add subplot labels
    subplotLabel(ax)
    
    # geneOntology(cmpNumb=24, geneAmount=50, geneValue="Overexpressed", axs=ax[0:6])
    geneOntology(cmpNumb=25, geneAmount=10, geneValue="Underexpressed", axs=ax[0:2])
    
    
    
    return f
