# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:38:45 2020

@author: Chaitanya Sampat

Sensitivity analysis of the PCNN wrt the input process paramters
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import csv

from SALib.sample import saltelli
from SALib.analyze import sobol
from keras.models import load_model

from HelperFunction import dataManipulation
from HelperFunction import plotterClass


# loading the previously trained and optimized  PCNN model 
pinnModel = 'PINNModel'
reload_pinnModel = load_model(pinnModel,compile=False)

# Defining the input problem size accoriding to the SALib data

samplesize = {
    'num_vars':6,
    'names': ['BA','LA','RPM','ID','SD','IP'],
    'bounds': [[800, 2000],
               [600, 1200],
               [100, 600],
               [0.075, 0.25],
               [200, 700],
               [0.1, 0.5]
            ]
}

parameter_vals = saltelli.sample(samplesize, 1000)
minVals = parameter_vals.min(axis=0)
maxVals = parameter_vals.max(axis=0)
maxVals[4] = 1000
norm_paramVals = np.zeros(parameter_vals.shape)


# normalizing inputs since the PCNN model takes only normalized inputs

for i in range(samplesize['num_vars']):
    for n in range(len(parameter_vals)):
        norm_paramVals[n,i] = (parameter_vals[n,i] - samplesize['bounds'][i][0]) /(samplesize['bounds'][i][1] - samplesize['bounds'][i][0])


# Predicting the output using PCNN

prediction_pinnModel = reload_pinnModel.predict(norm_paramVals)

a = np.array(prediction_pinnModel[0])
b = np.array(prediction_pinnModel[1])
d = np.zeros((len(a),len(b[0])+1))

for i in range(len(a)):
    d[i,0] = a[i]
    for j in range(1,len(b[0])+1):
        d[i,j] = b[i,j-1]

pinn_prediction_final = d
print(pinn_prediction_final.shape)

# Sobol dict for granule density
S_gd = sobol.analyze(samplesize,pinn_prediction_final[:,0])

# Sobol dict for individual sieves
S_bin1 = sobol.analyze(samplesize,pinn_prediction_final[:,1])
S_bin2 = sobol.analyze(samplesize,pinn_prediction_final[:,2])
S_bin3 = sobol.analyze(samplesize,pinn_prediction_final[:,3])
S_bin4 = sobol.analyze(samplesize,pinn_prediction_final[:,4])
S_bin5 = sobol.analyze(samplesize,pinn_prediction_final[:,5])
S_bin6 = sobol.analyze(samplesize,pinn_prediction_final[:,6])
S_bin7 = sobol.analyze(samplesize,pinn_prediction_final[:,7])
S_bin8 = sobol.analyze(samplesize,pinn_prediction_final[:,8])

print(S_gd['ST'])
dicNames = ['S_gd', 'S_bin1', 'S_bin2', 'S_bin3', 'S_bin4', 'S_bin5', 'S_bin6', 'S_bin7', 'S_bin8']
# with open('dict.csv','w',newline="") as csv_file:
#     writer = csv.writer(csv_file)
#     counter = 0
#     for dict1 in [S_gd, S_bin1, S_bin2, S_bin3, S_bin4, S_bin5, S_bin6, S_bin7, S_bin8]:
#         writer.writerow(dicNames[counter])
#         counter = counter+1
#         for key, value in dict1.items():
#             writer.writerow([key, value])


# creating a plotterClass object
plotObj = plotterClass.PlotterClass()

plotObj.sensPlot_2(S_gd,samplesize['names'],'Granule Density')
plotObj.sensPlot_2(S_bin1,samplesize['names'],'Bin1')
plotObj.sensPlot_2(S_bin2,samplesize['names'],'Bin2')
plotObj.sensPlot_2(S_bin3,samplesize['names'],'Bin3')
plotObj.sensPlot_2(S_bin4,samplesize['names'],'Bin4')
plotObj.sensPlot_2(S_bin5,samplesize['names'],'Bin5')
plotObj.sensPlot_2(S_bin6,samplesize['names'],'Bin6')
plotObj.sensPlot_2(S_bin7,samplesize['names'],'Bin7')
plotObj.sensPlot_2(S_bin8,samplesize['names'],'Bin8')
