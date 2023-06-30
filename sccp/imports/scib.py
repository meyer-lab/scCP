import anndata
from .scRNA import tensorFy

def import_scib_data(dataname):
    """Immune cell import from Theis lab: scib pipeline"""
    if dataname == "ImmuneHuman":
        annData = anndata.read_h5ad("/opt/andrew/scib/Immune_ALL_human.h5ad")
        print(annData)
        print(annData.to_df())
        print(annData.obs["species"])
        celltypes = annData.obs["final_annotation"].values.to_numpy()
        return tensorFy(annData, "batch"), celltypes
    elif dataname == "ImmuneHumanMouse":
        annData = anndata.read_h5ad("/opt/andrew/scib/Immune_ALL_hum_mou.h5ad")
        print(annData)
        celltypes = annData.obs["final_annotation"].values.to_numpy()
        return tensorFy(annData, "batch"), celltypes
    elif dataname == "Stimulation1":
        annData = anndata.read_h5ad("/opt/andrew/scib/sim1_1_norm.h5ad")
        print(annData)
        print(annData.obs["size_factors"])
        print(annData.obs["Group"])
        celltypes = annData.obs["Group"].values.to_numpy()
        return tensorFy(annData, "Batch"), celltypes
    elif dataname == "Stimulation2":
        annData = anndata.read_h5ad("/opt/andrew/scib/sim2_norm.h5ad")
        # print(annData.obs["Batch"].values)
        celltypes = annData.obs["Group"].values.to_numpy()
        return tensorFy(annData, "Batch"), celltypes
    elif dataname == "Pancreas":
        annData = anndata.read_h5ad("/opt/andrew/scib/human_pancreas_norm_complexBatch.h5ad")
        celltypes = annData.obs["celltype"].values.to_numpy()
        return tensorFy(annData, "tech"), celltypes
