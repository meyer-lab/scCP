"""
Lupus: Cell count/percentage for each cell type and status
"""

from anndata import read_h5ad
import seaborn as sns
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import cell_count_perc_df, rotate_xaxis
from matplotlib.axes import Axes
import anndata


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 6), (4, 2))

    # Add subplot labels
    subplotLabel(ax)
    
    
    types2 = {
	'CM' : ['CD14'
          'LYZ',
          'S100A8',
          'S100A9',
          'IL8'],
	'cDC1' : ['LGALGS2',
            'MS4A6A'],
	'cDC2' : ['HLA-DQA1',
            'HLA-DQA2',
            'HLA-DMB'],
	'nCM' : ['FCGR3A'],
	'pDC' : ['SERPINF1',
           'LILRA',
           'IRF8'],
	'B Naive' : ['TCL1A',
              'MS4A1'],
      'B Memory' : ['BANK1'],
      'B Plasma' : ['IGJ',
                      'CD79A',
                      'MZB1'],
      'T4 Naive' : ['CCD7',
                      'MAL',
                      'LEF1'],
      'T4 Reg' : ['ITGB1'],
      'T4 EM' : ['IL7R',
                   'RTKN1'],
      'T8 Naive' : ['CD8A',
                      'CD8B'],
      'T8 GZMK' : ['GZMK',
                     'GNLY'],
      'T8 GZMH' : ['GZMH'],
      'NK Bright' : ['GNLY'],
      'NK Dim' : ['NKG7'],
      'Prolif' : ['STMN1'],
	'Null' : ['RGS18'],
	}
    
    # types2 = {
	# 	'B cells' : [
	# 		'PXK',
	# 		'MS4A1',
	# 		'CD19',
	# 		'CD74',
	# 		'CD79A',
	# 		'BANK1',
	# 		'PTPRC',
	# 		'CR2'],
    #     'B Memory' : ["NPIB15",
    #                   "BACH2",
    #                   "IL7",
    #                   "NMBR",
    #                   "MS4A1",
    #                   "MBL2",
    #                   "LY86"],
    #     'B Naive' : ["P2RX5",
    #                   "SIK1",
    #                   "SLC12A1",
    #                   "SELL",
    #                   "RALGPS2",
    #                   "PTPRCAP",
    #                   "PSG2"],
    #     'Basophils' : ["CCL4",
    #                   "NPL",
    #                   "WRN",
    #                   "NFIL3",
    #                   "TEC",
    #                   "OTUB2",
    #                   "FAR2"],
    #     'DCs' : ["ITGAX",
    #                   "ZBTB46",
    #                   "LAMP3",
    #                   "CXCR1",
    #                   "ITGAM",
    #                   "FCER1A",
    #                   "IL6"],
    #     'Eosinophils' : ["CSF2",
    #                   "EPX",
    #                   "SIGLEC8",
    #                   "CCL5",
    #                   "IKZF2",
    #                   "CPA3",
    #                   "PRPG2"],
    #     'Gamma T Cells' : ["S100B",
    #                   "TUBB",
    #                   "TRGJP2",
    #                   "STMN1",
    #                   "TRGV9",
    #                   "CCL5",
    #                   "HMGB2"],
	#     'Macrophages' : ["CD68",
    #                   "FCGR1",
    #                   "NAAA",
    #                   "JAML",
    #                   "TYROBP",
    #                   "LYZ2",
    #                   "H2-DMA"],
	#     'Mast Cells' : ["SLC29A1",
    #                   "KIT",
    #                   "LTC4S",
    #                   "TPSAB1",
    #                   "IL1RL1",
    #                   "HDC",
    #                   "TPSB2"],
	#     'Megakaryocytes' : ["PLK3",
    #                   "PROX1",
    #                   "SYP",
    #                   "TSPAN9",
    #                   "RGS18",
    #                   "GATA2",
    #                   "VWF"],
    #     'Monocytes' : ["APOBEC3A",
    #                   "LYZ",
    #                   "CD14",
    #                   "CFP",
    #                   "HLA-DRA",
    #                   "S100A9",
    #                   "S100A8",
    #                   "CSF3R"],
	# 	'Myeloid Suppressors' : ["S100A4",
    #                   "S100A9",
    #                   "ICAM1",
    #                   "S100A8",
    #                   "ITGAM",
    #                   "LY6G",
    #                   "GR1"],
	#     'NKT' : ["IL12RB2",
    #                   "NCAM1",
    #                   "GATA3",
    #                   "CD44",
    #                   "IL2RB",
    #                   "CXCR4",
    #                   "SLAMF7"],
	#     'Neutrophils' : ["CSF3R",
    #                   "LY6G",
    #                   "S100A8",
    #                   "TREM1",
    #                   "IL1R2",
    #                   "CFP",
    #                   "ADAM8"],
	#     'NK' : ["NKG7",
    #                   "GNLY",
    #                   "KLRD1",
    #                   "KLRF1",
    #                   "NCR1",
    #                   "DOCK2",
    #                   "GZMA"],
	#     'Nuocytes' : ["IL1RL1",
    #                   "ICOS",
    #                   "IL17RB",
    #                   "IL7R",
    #                   "CRLF2",
    #                   "ARG1",
    #                   "GATA3"],
    # 	'Plasma Cells' : ["MZB1",
    #                   "IGHG1",
    #                   "SPAG4",
    #                   "TGM5",
    #                   "SIK1",
    #                   "RPL3P7",
    #                   "RGS13"],
    #     'Plasmacytoid DCs' : ["BST2",
    #                   "CLEC4C",
    #                   "MAP3K2",
    #                   "KLK1",
    #                   "CMAH",
    #                   "TRADD",
    #                   "LILRA4"],
    #     'T Cells' : ["TRBC2",
    #                   "CD3D",
    #                   "CD3G",
    #                   "CD3E",
    #                   "LTB",
    #                   "IL7R",
    #                   "LEF1"],
    #     'Cytotoxic T' : ["TRAC",
    #                   "CD8A",
    #                   "GZMB",
    #                   "CD2",
    #                   "CD27",
    #                   "CD5",
    #                   "CD27"],
    #     'Follicular T' : ["ICOS",
    #                   "PDCD1",
    #                   "BCL6",
    #                   "CXCR5",
    #                   "CD200",
    #                   "P2RX7",
    #                   "CD3D"],
    #     'Helper T' : ["CCR4",
    #                   "CD4",
    #                   "IL13",
    #                   "CD28",
    #                   "CD3G",
    #                   "IL2",
    #                   "CCR6"],
    #     'Memory T' : ["CCR7",
    #                   "CD2",
    #                   "PTPRC",
    #                   "CD28",
    #                   "LEF1",
    #                   "S100A8",
    #                   "GIMAP4"],
    #     'Follicular T' : ["IKZF2",
    #                   "FOXP3",
    #                   "CCR4",
    #                   "ENTPD1",
    #                   "IL2RA",
    #                   "ITGAE",
    #                   "TNFRSF4",
    #                   "CTLA4"],
	# }
    for i in types2:
        k = types2[i]
        print(i)
        for j in k:
            print(j)
        
        

    # X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")

    # celltype = ["Cell Type", "Cell Type2", "leiden"]
    # label = ["Cell Type Percentage", "Cell Count"]
    # plot = 0

    # for i in range(len(celltype)):
    #     for j in range(len(label)):
    #         df = cell_count_perc_df(X, celltype=celltype[i], status=True)
    #         sns.boxplot(
    #             data=df,
    #             x="Cell Type",
    #             y=label[j],
    #             hue="SLE_status",
    #             showfliers=False,
    #             ax=ax[plot],
    #         )
    #         rotate_xaxis(ax[plot])
    #         plot += 1

    # plot_cell_count_status(X, ax[6])
    # f.delaxes(ax[7])

    return f


def plot_cell_count_status(X: anndata.AnnData, ax: Axes):
    """Plots overall cell count for SLE and healthy patients"""
    df = X.obs[["SLE_status", "Condition"]].reset_index(drop=True)
    dfCond = (
        df.groupby(["Condition", "SLE_status"], observed=True)
        .size()
        .reset_index(name="Cell Count")
    )

    sns.boxplot(
        data=dfCond,
        x="SLE_status",
        y="Cell Count",
        hue="SLE_status",
        showfliers=False,
        ax=ax,
    )

    return
