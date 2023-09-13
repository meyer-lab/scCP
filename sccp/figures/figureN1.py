"""
Plot R2X values when dropping a cell type across a drug condition.
"""
import os
from .common import (subplotLabel, 
getSetup)
import seaborn as sns
import pandas as pd


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((13, 13), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    data = {
        'Cell Type': [],
        'Drug': [],
        'R2X': []
    }

    data_dir = './sccp/data/r2x_dropping'
    for file in os.listdir(data_dir):
        r2x_vals = ''
        with open(os.path.join(data_dir, file), 'r') as fp:
                for i in fp.readlines():
                    r2x_vals = i
        r2x_vals = eval(r2x_vals)
        for cell_type, condition in r2x_vals.keys():
            data['Cell Type'].append(cell_type)
            data['Drug'].append(condition)
            data['R2X'].append(r2x_vals[cell_type, condition])

    data = pd.DataFrame(data)
    sns.barplot(x='Cell Type', y='R2X', hue='Drug', data=data, ax=ax[0])   
    
    return f