import xarray as xa
import numpy as np
import pandas as pd
from collections import Counter
import os
from os.path import join

path_here = os.path.dirname(os.path.dirname(__file__))


def CoH_xarray(allmarkers=True, saveXA=False):
    """Reducing CoH dataframe with experiments consistent across patients into XA"""
    
    if saveXA == True:
        scDF = (
            pd.read_csv("/opt/andrew/CoH_Flow_SC_NoMissingnessV1.csv")
            .reset_index(drop=True)
            .drop("Unnamed: 0", axis=1)
            )
        
        scDF.drop(columns=["Cell", "Time"], axis=1, inplace=True)
        
        status_DF = (
            pd.read_csv("sccp/data/CoH_Patient_Status.csv")
            .reset_index(drop=True)
            .drop("Unnamed: 0", axis=1)
        )

        # Renaming patients
        healthy = 0
        bc = 0
        for i, stat in enumerate(status_DF["Patient"].values):
            statDF = status_DF.loc[status_DF["Patient"] == stat]
            if statDF["Status"].values == "Healthy":
                healthy += 1
                scDF = scDF.replace(stat, "Healthy-" + str(healthy))
            else:
                bc += 1
                scDF = scDF.replace(stat, "BC-" + str(bc))

        if allmarkers == True:
            marker_dict = marker_dict_surface
        else:
            marker_dict = marker_dict_stat

        scDF.sort_values(by=["Treatment","Patient"], inplace=True)
        assert np.all(np.isfinite(scDF[marker_dict].to_numpy()))

        for mark in marker_dict:
            scDF = scDF[scDF[mark] < scDF[mark].quantile(0.995)]

        # Normalization of markers
        scDF[justmark] = scDF.groupby(by=["Patient"])[justmark].transform(
            lambda x: x / np.mean(x, axis=0)
        )
        scDF[marker_dict_stat] /= np.mean(scDF[marker_dict_stat], axis=0)

        experimentcells = scDF.groupby(by=["Treatment", "Patient"]).size().values
        scDF["Cell"] = np.concatenate([np.arange(int(cnt)) for cnt in experimentcells])

        # Changing to Xarray
        CoHxa = scDF.set_index(["Cell", "Treatment", "Patient"]).to_xarray()
        celltypeXA = CoHxa["CellType"]
        CoHxa = CoHxa.drop_vars(["CellType"])
        CoHxa = CoHxa[marker_dict].to_array(dim="Marker")
        CoHxa.values = np.nan_to_num(CoHxa.values)
            
        CoH_XA = CoHxa.sel(Patient=patients).transpose()
        celltypeXA = celltypeXA.sel(Patient=patients).transpose()
        # Final Xarray has dimensions [Patient, Treatment, Cell, Marker]
            
        if allmarkers == True:
            CoH_XA.to_netcdf(join(path_here, "data/CoH_flowXA_AllMarkers.nc"))
            celltypeXA.to_netcdf(join(path_here, "data/CoH_celltypeXA_AllMarkers.nc"))
        else:
            CoH_XA.to_netcdf(join(path_here, "data/CoH_flowXA_OnlySTAT.nc"))
            celltypeXA.to_netcdf(join(path_here, "data/CoH_celltypeXA_OnlySTAT.nc"))
            
    else:
        if allmarkers == True:
            CoH_XA = xa.open_dataarray(join(path_here, "/opt/andrew/CoH_flowXA_AllMarkers.nc"))
            celltypeXA = xa.open_dataarray(join(path_here, "/opt/andrew/CoH_celltypeXA_AllMarkers.nc"))
        else:    
            CoH_XA = xa.open_dataarray(join(path_here, "/opt/andrew/CoH_flowXA_OnlySTAT.nc"))
            celltypeXA = xa.open_dataarray(join(path_here, "/opt/andrew/CoH_celltypeXA_OnlySTAT.nc"))

    return CoH_XA, celltypeXA


patients = [
    "Healthy-1",
    "Healthy-2",
    "Healthy-3",
    "Healthy-4",
    "Healthy-5",
    "Healthy-6",
    "Healthy-7",
    "Healthy-8",
    "Healthy-9",
    "Healthy-10",
    "Healthy-11",
    "Healthy-12",
    "Healthy-13",
    "Healthy-14",
    "Healthy-15",
    "Healthy-16",
    "Healthy-17",
    "Healthy-18",
    "Healthy-19",
    "Healthy-20",
    "Healthy-21",
    "Healthy-22",
    "BC-1",
    "BC-2",
    "BC-3",
    "BC-4",
    "BC-5",
    "BC-6",
    "BC-7",
    "BC-8",
    "BC-9",
    "BC-10",
    "BC-11",
    "BC-12",
    "BC-13",
    "BC-14",
]


marker_dict_all = [
    "SSC-H",
    "SSC-A",
    "FSC-H",
    "FSC-A",
    "SSC-B-H",
    "SSC-B-A",
    "CD45RA",
    "Live/Dead",
    "CD4",
    "CD16",
    "CD8",
    "pSTAT6",
    "PD-L1",
    "CD3",
    "PD-1",
    "CD14",
    "CD33",
    "CD27",
    "pSTAT3",
    "pSTAT1",
    "pSmad1-2",
    "FoxP3",
    "pSTAT5",
    "pSTAT4",
    "CD20",
]

marker_dict_surface = [
    "CD45RA",
    "CD4",
    "CD16",
    "CD8",
    "pSTAT6",
    "PD-L1",
    "CD3",
    "PD-1",
    "CD14",
    "CD33",
    "CD27",
    "pSTAT3",
    "pSTAT1",
    "pSmad1-2",
    "FoxP3",
    "pSTAT5",
    "pSTAT4",
    "CD20",
]

marker_dict_stat = ["pSTAT6", "pSTAT3", "pSTAT1", "pSmad1-2", "pSTAT5", "pSTAT4"]

justmark = list((Counter(marker_dict_surface) - Counter(marker_dict_stat)).elements())
