import anndata
from .scRNA import tensorFy

def import_scib_data(dataname):
    """Immune cell import from Theis lab: scib pipeline"""
    if dataname == "ImmuneHuman":
        annData = anndata.read_h5ad("/opt/andrew/scib/Immune_ALL_human.h5ad")
    elif dataname == "ImmuneHumanMouse":
        annData = anndata.read_h5ad("/opt/andrew/scib/Immune_ALL_hum_mou.h5ad")
    elif dataname == "Stimulation1":
        annData = anndata.read_h5ad("/opt/andrew/scib/sim1_1_norm.h5ad")
    elif dataname == "Stimulation2":
        annData = anndata.read_h5ad("/opt/andrew/scib/sim2_norm.h5ad")
    elif dataname == "Pancreas":
        annData = anndata.read_h5ad("/opt/andrew/scib/human_pancreas_norm_complexBatch.h5ad")
    
    return tensorFy(annData, "batch")