"""Investigating metrics for datasets"""
from .common import (subplotLabel, getSetup, 
   plotMetricSCIB)
from ..imports.scib import  import_scib_metrics
from pandas.plotting import parallel_coordinates as pc
import matplotlib.pyplot as plt

def makeFigure():
   """Get a list of the axis objects and create a figure."""
   # Get list of axis objects
   ax, f = getSetup((25, 25), (4, 2))

   # Add subplot labels
   subplotLabel(ax)
   
   metricsDF, sheetName, metrics = import_scib_metrics()
   
   #  print(metricsDF.loc[(metricsDF["Dataset"] == "pancreas") & (metricsDF["Metric"] == "PCR batch")])

   # pc(metricsDF)

   # datasetDF = metricsDF.loc[metricsDF["Dataset"] == "pancreas"]
   # print(datasetDF)
   # datasetDF = datasetDF.pivot(index=columns="Method", values=f"Value")
   # print(datasetDF)
   # pc(datasetDF, "Value", colormap =plt.get_cmap("Set2"), ax=ax[0])
   # sns.stripplot(data=datasetDF, x="Metric", y = "Value", hue="Method", ax=axs[i])
   
   #  plotMetricSCIB(metricsDF, sheetName, ax[0:7])


   return f