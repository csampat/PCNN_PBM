# -*- coding: utf-8 -*-
"""
Created on Tue May 5 10:57:45 2020

@author: Chaitanya Sampat

Test prediction of models using previously trained PCNN models
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K

from tensorflow import keras
from keras.layers import Dense, Input, Concatenate
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD

from HelperFunction import dataManipulation
from HelperFunction import plotterClass


def predictAndPlot(dataSet, actualVal, sieveCut, testIdx, modelNames, frac_retrain=0.0):
    #creating objects for helper functions
    plotObj = plotterClass.PlotterClass()

    simModel = modelNames[0]
    stdeModel = modelNames[1]
    stsmaxModel = modelNames[2]
    pinnModel = modelNames[3]

    # reloading models and prediction data

    '''
    Compile is False since custom loss function was used. 
    If you need to retrain the model consider defining the custom loss function 
    Similar syntax as the one found in dataManipulation.py should be used
    '''

    reload_simModel = load_model(simModel,compile=False)
    reload_stdeModel = load_model(stdeModel,compile=False)
    reload_stsmaxModel = load_model(stsmaxModel,compile=False)
    reload_pinnModel = load_model(pinnModel,compile=False)

    # Prediction the labels from data file data file 
    
    prediction_simModel = reload_simModel.predict(dataSet)
    prediction_stdeModel = reload_stdeModel.predict(dataSet)
    prediction_stsmaxModel = reload_stsmaxModel.predict(dataSet)
    prediction_pinnModel = reload_pinnModel.predict(dataSet)

    # flattening and converting all prediction to the shape (len(labels),len(labels.keys()))

    prediction_simModel = prediction_simModel.flatten()
    prediction_stdeModel = prediction_stdeModel.flatten()
    prediction_stsmaxModel = prediction_stsmaxModel.flatten()
    
    prediction_simModel_conv = np.reshape(np.ravel(np.array(prediction_simModel)),(len(actualVal),9))
    prediction_stdeModel_conv = np.reshape(np.ravel(np.array(prediction_stdeModel)),(len(actualVal),9))
    prediction_stsmaxModel_conv = np.reshape(np.ravel(np.array(prediction_stsmaxModel)),(len(actualVal),9))
    
    a = np.array(prediction_pinnModel[0])
    b = np.array(prediction_pinnModel[1])
    d = np.zeros((len(a),len(b[0])+1))

    for i in range(len(a)):
        d[i,0] = a[i]
        for j in range(1,len(b[0])+1):
            d[i,j] = b[i,j-1]

    pinn_prediction_final = d.flatten()
    
    pinn_prediction_conv = np.reshape(np.ravel(pinn_prediction_final),(len(actualVal),9))
    

    pinn_prediction_conv = np.reshape(np.ravel(pinn_prediction_final),(len(actualVal),9))
    
    # plotting comparison plots

    plotObj.expDataPlot_comparen([prediction_simModel_conv,prediction_stdeModel_conv,prediction_stsmaxModel_conv,pinn_prediction_conv],actualVal,testIdx,sieveCut,[simModel,stdeModel,stsmaxModel,pinnModel])



def main():
    noGrowthFile = pd.read_csv('./PBM_noGrowth.csv')
    fastGrowthFile = pd.read_csv('./PBM_fastGrowth.csv')
    stdeViolationFile = pd.read_csv('./PBM_stdeViolation.csv')
    sieveCut = [150,250,350,600,850,1000,1400]
    testIdx = [20,50,100,250,500,600]
    frac_retrain = 0.0
    #creating objects for helper functions
    # plotObj = plotterClass.PlotterClass()
    hFun_noGrowth = dataManipulation.HelperFunction(noGrowthFile,sieveCut,frac_retrain)
    
    # splitting and normalizing dataset into data and labels (no internal split for train/test)
    dataSet_noGrowth, actualVal_noGrowth = hFun_noGrowth.normedDataLabelSplit(noGrowthFile)

    # Slow growth
    hFun_fastGrowth = dataManipulation.HelperFunction(fastGrowthFile,sieveCut,frac_retrain)
    dataSet_fastGrowth, actualVal_fastGrowth = hFun_fastGrowth.normedDataLabelSplit(fastGrowthFile)
    
    #stde violation
    hFun_stdeVio = dataManipulation.HelperFunction(stdeViolationFile,sieveCut,frac_retrain)
    dataSet_stdeVio, actualVal_stdeVio = hFun_stdeVio.normedDataLabelSplit(stdeViolationFile)

    
    

    # loading models with their corresponding saved names
    simModel = 'AllDataANNModel'
    stdeModel = 'StDeRefinedDataModel'
    stsmaxModel = 'StDeSmaxRefinedDataModel'
    pinnModel = 'PINNModel'
    
    # no Growth file
    predictAndPlot(dataSet_noGrowth, actualVal_noGrowth, sieveCut, testIdx, [simModel,stdeModel,stsmaxModel,pinnModel])
    # Slow Growth file
    predictAndPlot(dataSet_fastGrowth, actualVal_fastGrowth, sieveCut, testIdx, [simModel,stdeModel,stsmaxModel,pinnModel])
    # StDe violated file
    predictAndPlot(dataSet_stdeVio, actualVal_stdeVio, sieveCut, testIdx, [simModel,stdeModel,stsmaxModel,pinnModel])

    plt.show()

if __name__ == "__main__":
    main()

