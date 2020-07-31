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
import talos

from tensorflow import keras
from keras.layers import Dense, Input, Concatenate
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, Adam
from talos.model.normalizers import lr_normalizer

from HelperFunction import dataManipulation
from HelperFunction import plotterClass

sieveCut = [150,250,350,600,850,1000,1400]
dataFile = pd.read_csv('./PBM_simData_solidDensity.csv')

# Creating class objects for PINN model
pinnMod = dataManipulation.HelperFunction(dataFile, sieveCut, frac=0.2)
pinn_train_data, pinn_train_labels, pinn_test_data, pinn_test_labels = pinnMod.normed_train_dataset, pinnMod.train_labels, pinnMod.normed_test_dataset, pinnMod.test_labels
# pinn_train_data = pinn_train_data.drop(columns=['d50','StDe','Smax'])
# pinn_test_data = pinn_test_data.drop(columns=['d50','StDe','Smax'])
PINN_name = 'PINN model'

hyperparameters = {
    "epochs": [10,25,50,100,200,300,400], 
    "activation": ['tanh','relu'],
    "last_activation": ['linear'],
    "nNodes": [4,8,12,16,24,32],
    "learningRate": [0.3,0.03,0.003,0.0003],
    "dropRate": [0.0,0.1,0.2,0.4,0.5],
    "regCons": [0.0,0.01,0.001,0.1],
    "patienceModel": [10],
    "optimizer": [Adam,SGD]
}

scan_object = talos.Scan(pinn_train_data,pinn_train_labels,params=hyperparameters,model=pinnMod.hyperparameter_Scan,experiment_name='PINN_model',x_val=pinn_test_data,y_val=pinn_test_labels)


# accessing the results data frame
scan_object.data.head()

# accessing epoch entropy values for each round
scan_object.learning_entropy

# access the summary details
scan_object.details
# accessing the saved models
scan_object.saved_models

# accessing the saved weights for models
scan_object.saved_weights

# use Scan object as input
analyze_object = talos.Analyze(scan_object)

# access the dataframe with the results
analyze_object.data

# get the number of rounds in the Scan
analyze_object.rounds()

# get the highest result for any metric
analyze_object.low('val_loss')

# get the best paramaters
analyze_object.best_params('val_loss', ['out1_mse', 'out2_mse', 'loss', 'val_loss'])

# get correlation for hyperparameters against a metric
analyze_object.correlate('val_loss', ['out1_mse', 'out2_mse', 'loss', 'val_loss'])


# line plot
analyze_object.plot_line('val_loss')

# up to two dimensional kernel density estimator
analyze_object.plot_kde('val_loss')

# a simple histogram
analyze_object.plot_hist('val_loss', bins=50)

# heatmap correlation
analyze_object.plot_corr('val_loss', ['out1_mse', 'out2_mse', 'loss', 'val_loss'])

# a four dimensional bar grid
analyze_object.plot_bars('epochs', 'val_loss', 'activation', 'optimizer')

plt.show()