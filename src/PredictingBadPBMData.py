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

from HelperFunction import dataManipulation
from HelperFunction import plotterClass




def main():
    noGrowthFile = pd.read_csv('./PBM_noGrowth.csv')
    sieveCut = [150,250,350,600,850,1000,1400]

    frac_retrain = 0.0
    plotObj = plotterClass.PlotterClass()
    hFunObj = dataManipulation.HelperFunction(noGrowthFile,sieveCut,frac_retrain)

        
    normed_train_dataset, train_labels, normed_test_dataset, test_labels = hFunObj.normed_train_dataset, hFunObj.train_labels, hFunObj.normed_test_dataset, hFunObj.test_labels
    model1 = 'AllData_2000'
    recons_model = load_model(model1)
    test_predictions_sm = recons_model.predict(normed_test_dataset)

    test_conv_sm = np.array(test_predictions_sm)
    test_conv_sm = np.reshape(np.ravel(test_conv_sm),(len(test_labels),9))
    testIdx = [20,50]

    plotObj.expDataPlot_comparen([test_conv_sm],test_labels,testIdx,sieveCut,[model1])
    plt.show()

if __name__ == "__main__":
    main()

