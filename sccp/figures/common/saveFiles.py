import numpy as np
import os
from os.path import join

import pickle



path_here = os.path.dirname(os.path.dirname(__file__))

def savePf2(weight, factors, projs, dataName: str):
    """Saves weight factors and projections for one dataset for a component"""
    rank = len(weight)
    np.save(join(path_here, "data/"+dataName+"/"+dataName+"_WeightCmp"+str(rank)+".npy"), weight)
    factor = ["A", "B", "C"]
    for i in range(3):
        np.save(join(path_here, "data/"+dataName+"/"+dataName+"_Factor"+str(factor[i])+"Cmp"+str(rank)+ ".npy"), factors[i])
    np.save(join(path_here, "data/"+dataName+"/"+dataName+"_ProjCmp"+str(rank)+".npy"), np.concatenate(projs, axis=0))

    
def openPf2(rank: int, dataName: str, optProjs = False):
    """Opens weight factors and projections for one dataset for a component as numpy arrays"""
    weight = np.load(join(path_here, "data/"+dataName+"/"+dataName+"_WeightCmp"+str(rank)+".npy"), allow_pickle=True)
    factors = [np.load(join(path_here, "data/"+dataName+"/"+dataName+"_FactorACmp"+str(rank)+ ".npy"), allow_pickle=True),
               np.load(join(path_here, "data/"+dataName+"/"+dataName+"_FactorBCmp"+str(rank)+ ".npy"), allow_pickle=True),
               np.load(join(path_here, "data/"+dataName+"/"+dataName+"_FactorCCmp"+str(rank)+ ".npy"), allow_pickle=True)]
        
    if optProjs is False:
        projs = np.load(join(path_here, "data/"+dataName+"/"+dataName+"_ProjCmp"+str(rank)+".npy"), allow_pickle=True)
    else:
        projs = np.load(join(path_here, "/opt/andrew/"+dataName+"/"+dataName+"_ProjCmp"+str(rank)+".npy"), allow_pickle=True)
        
    return weight, factors, projs

def saveUMAP(fit_points, rank:int, dataName: str):
    """Saves UMAP points locally, large files uploaded manually to opt"""
    f_name = join(path_here, "data/"+dataName+"/"+dataName+"_UMAPCmp"+str(rank)+".sav")
    pickle.dump(fit_points, open(f_name, 'wb'))

def openUMAP(rank: int, dataName: str, opt = True):
    """Opens UMAP points for plotting, defaults to using the opt folder (for big files)"""
    if opt == True:
        f_name = join(path_here, "/opt/andrew/"+dataName+"/"+dataName+"_UMAPCmp"+str(rank)+".sav")
    else:
        f_name = join(path_here, "data/"+dataName+"/"+dataName+"_UMAPCmp"+str(rank)+".sav")
    return pickle.load((open(f_name, 'rb')))