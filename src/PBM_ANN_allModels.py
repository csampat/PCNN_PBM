# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:53:13 2020

@author: Chaitanya Sampat
"""


# recreating run file for easier reading


import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from tensorflow import keras

from HelperFunction import dataManipulation
from HelperFunction import plotterClass

start_time = time.time()
sieveCut = [150,250,350,600,850,1000,1400]

# Datafile without sieves
dataFile = pd.read_csv('./PBM_simData_solidDensity.csv')
dataFileWithDensity = pd.read_csv('./PBM_simData_solidDensity.csv')

# Creating class objects that are 
simMod = dataManipulation.HelperFunction(dataFile,sieveCut)
stref = dataManipulation.HelperFunction(dataFile,sieveCut)
stsmaxred = dataManipulation.HelperFunction(dataFile,sieveCut)
densMod = dataManipulation.HelperFunction(dataFileWithDensity,sieveCut)
plotObj = plotterClass.PlotterClass()

# defining number of output layers
nOuput = 9
nOutDensity = 9

# creating separate datafiles so as to not affect data

normed_train_dataset, train_labels, normed_test_dataset, test_labels = simMod.normed_train_dataset, simMod.train_labels, simMod.normed_test_dataset, simMod.test_labels
red_dataFile = stref.stdeBasedDataRemoval()
red_normed_train_dataset, red_train_labels, red_norm_test_dataset, red_test_labels = stref.normedDataSplitWithDensity(red_dataFile)
redsmax_dataFile = stsmaxred.stdesmaxDataRemoval()
redsmax_normed_train_dataset, redsmax_train_labels, redsmax_norm_test_dataset, redsmax_test_labels = stsmaxred.normedDataSplitWithDensity(redsmax_dataFile)
dens_normed_train_dataset, dens_train_labels, dens_normed_test_dataset, dens_test_labels = densMod.normed_train_dataset, densMod.train_labels, densMod.normed_test_dataset, densMod.test_labels


# Define different types of models used
model1 = 'All data '
model2 = 'StDE refined  '
model3 = 'StDe + Smax refined '
# model4 = 'PINN with StDe '
# model5 = 'PINN with StDe and Smax '
# model6 = 'Sieve + Density '
model7 = 'PINN with L2'
model8 = 'PINN with L1'


patience_model = 5
EPOCHS = 1

# simple model

simpleModel, history = simMod.build_train_model_2intLayers(normed_train_dataset,train_labels,patience_model,nOuput,16,'relu',EPOCHS)
m1Time = time.time()

# # # # # Data refined models
strefmodel, stref_history = stref.build_train_model_2intLayers(red_normed_train_dataset,red_train_labels,patience_model,nOuput,16,'relu',EPOCHS)
m2Time = time.time()


smaxref_model, smaxref_history = stsmaxred.build_train_model_2intLayers(redsmax_normed_train_dataset,redsmax_train_labels,patience_model,nOuput,16,'relu',EPOCHS)
m3Time = time.time()


'''
########### OLD MODELS ##############
# PINN with StDe
# PINNstde_model, PINNstde_history = simMod.build_train_PINNstde(normed_train_dataset,train_labels,patience_model,nOuput,16,'relu',EPOCHS)
# m4Time = time.time() 

# PINNboth_model, PINNboth_history = simMod.build_train_PINNStdeSmax(normed_train_dataset,train_labels,patience_model,nOuput,16,'relu',EPOCHS)
# m5Time = time.time()

# models with density as output

# PINNdens_model_32, PINNdens_history_32 = densMod.build_train_PINN_mul(dens_normed_train_dat  aset,dens_train_labels,patience_model,nOutDensity,32,'relu',EPOCHS)
# m6Time = time.time()

# # evaluate the PINN model
# # loss_PINNstde, mae_PINNstde, mse_PINN = PINNstde_model.evaluate(normed_test_dataset, test_labels, verbose=0)
# # test_predictions_PINNstde = PINNstde_model.predict(normed_test_dataset).flatten()


# # # evaluate the PINN with stde and smax errors model
# # loss_PINNboth, mae_PINNboth, mse_PINNboth = PINNboth_model.evaluate(normed_test_dataset, test_labels, verbose=0)
# # test_predictions_PINNboth = PINNboth_model.predict(normed_test_dataset).flatten()
'''

EPOCHS = 100

###### NEW PINN #############
PINNdens_model, PINNdens_history = densMod.build_train_PINN_l2_dropout(dens_normed_train_dataset,dens_train_labels,patience_model,nOutDensity,16,'sigmoid',EPOCHS)
m7Time = time.time()


PINNdens_model_32, PINNdens_history_32 = densMod.build_train_PINN_l1_dropout(dens_normed_train_dataset,dens_train_labels,patience_model,nOutDensity,16,'sigmoid',EPOCHS)
m8Time = time.time()

# # evaluate simple model
loss_sm, mae_sm, mse_sm = simpleModel.evaluate(normed_test_dataset, test_labels, verbose=0)
test_predictions_sm = simpleModel.predict(normed_test_dataset).flatten()



# # use stde refined model to predict all data
loss_stref, mae_stref, mse_stref = strefmodel.evaluate(normed_test_dataset, test_labels, verbose=0)
test_predictions_stref = strefmodel.predict(normed_test_dataset).flatten()

# # use stde smax model to predict all data
loss_smax, mae_smax, mse_smax = smaxref_model.evaluate(normed_test_dataset, test_labels, verbose=0)
test_predictions_smax = smaxref_model.predict(normed_test_dataset).flatten()



###########
dens_test_labels1 = dens_test_labels.copy()

# use density PINN to predict data
label1 = dens_test_labels1['Granule_density']
labels = ['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']

label2 = pd.DataFrame([dens_test_labels1.pop(i) for i in labels]).T
###########

loss_densPINN, loss3, loss4, mae_densPINN, mse_densPINN, mae4, mse4 = PINNdens_model.evaluate(dens_normed_test_dataset, [label1,label2], verbose=0)
test_predictions_densPINN = PINNdens_model.predict(dens_normed_test_dataset)
test_predictions_densPINN = np.concatenate((test_predictions_densPINN[0].flatten(),test_predictions_densPINN[1].flatten()))

# use density PINN trained data model
loss_dens, loss3_dens, loss4_dens, mae_dens, mse_dens, mae4_dens, mse4_dens = PINNdens_model_32.evaluate(dens_normed_test_dataset, [label1,label2], verbose=0)
test_predictions_dens = PINNdens_model_32.predict(dens_normed_test_dataset)
test_predictions_dens = np.concatenate((test_predictions_dens[0].flatten(),test_predictions_dens[1].flatten()))

# # use all data model to predict small data

r2model_sm = simMod.calculateR2(test_predictions_sm, test_labels)
r2model_stref = stref.calculateR2(test_predictions_stref, test_labels)
r2model_smax = stsmaxred.calculateR2(test_predictions_smax, test_labels)
# r2model_PINNstde = simMod.calculateR2(test_predictions_PINNstde, test_labels)
# r2model_PINNboth = simMod.calculateR2(test_predictions_PINNboth, test_labels)
r2model_dens = simMod.calculateR2(test_predictions_dens, dens_test_labels)
r2model_densPINN = densMod.calculateR2(test_predictions_densPINN, dens_test_labels)

# use PINN to predict data


print("\nTest set Loss(MSE), MAE and R^2 of ", model1 ,"is", loss_sm, mae_sm, r2model_sm)
print("Test set Loss(MSE), MAE and R^2 of ", model2 ,"is",  loss_stref, mae_stref, r2model_stref)
print("Test set Loss(MSE), MAE and R^2 of ", model3 ,"is",  loss_smax, mae_smax, r2model_smax)
# print("Test set Loss(MSE), MAE and R^2 of ", model4 ,"is",  loss_PINNstde, mae_PINNstde, r2model_PINNstde)
# print("Test set Loss(MSE), MAE and R^2 of ", model5 ,"is",  loss_PINNboth, mae_PINNboth, r2model_PINNboth)
print("Test set Loss(MSE), MAE and R^2 of ", model7 ,"is",  loss_densPINN, mae_densPINN, r2model_densPINN)
print("Test set Loss(MSE), MAE and R^2 of ", model8 ,"is",  loss_dens, mae_dens, r2model_dens)


# print("All data set Loss, MAE and MSE of ", model2 ,"is",  loss_all, mae_all, mse_all)
# print("Small data set Loss, MAE and MSE of ", model2 ,"is",  loss_small, mae_small, mse_small)

#### saving model


simpleModel.save('AllData_2000')
strefmodel.save('StDErefinedModel_2000')
smaxref_model.save('StDE_SMax_refinedModel_2000')
PINNdens_model.save('PINN_16_2000')
PINNdens_model_32.save('PINN_dropout_2000')
print("--------All models were saved successfully--------")

# All plots start here
# comparing 3 models
plt.figure()
plotObj.history_plotter_comparen_noval([history, stref_history,smaxref_history,PINNdens_history,PINNdens_history_32], 'loss', [model1, model2, model3,model7,model8])

plt.figure()
plotObj.parityPlot_dens(test_labels, test_predictions_densPINN, model7)

plt.figure()
plotObj.parityPlot_dens(test_labels, test_predictions_dens, model8)

plt.figure()
plotObj.parityPlot_dens(test_labels, test_predictions_densPINN, model7)

plt.figure()
plotObj.parityPlot_dens(test_labels, test_predictions_dens, model8)

'''

plt.figure()
plotObj.parityPlot(test_labels, test_predictions_sm,model1)
plt.figure()
plotObj.parityPlot(test_labels, test_predictions_stref, model2)
# plt.figure()
# plotObj.parityPlot(test_labels, test_predictions_dens, model8)
plt.figure()
plotObj.parityPlot(test_labels, test_predictions_densPINN, model7)
plt.figure()
plotObj.parityPlot(test_labels, test_predictions_smax, model3)

plt.figure()
plotObj.parityPlot_dens(test_labels, test_predictions_densPINN, model7)


plt.figure()
plotObj.parityPlot_dens(test_labels, test_predictions_sm,model1)
plt.figure()
plotObj.parityPlot_dens(test_labels, test_predictions_stref, model2)
# plt.figure()
# plotObj.parityPlot(test_labels, test_predictions_dens, model8)


plt.figure()
plotObj.parityPlot_dens(test_labels, test_predictions_smax, model3)

test_conv_sm = np.array(test_predictions_sm)
test_conv_sm = np.reshape(np.ravel(test_conv_sm),(len(test_labels),9))

test_conv_stref = np.array(test_predictions_stref)
test_conv_stref = np.reshape(np.ravel(test_conv_stref),(len(test_labels),9))

# test_conv_dens = np.array(test_predictions_dens)
# test_conv_dens = np.reshape(np.ravel(test_conv_dens),(len(dens_test_labels),9))

test_conv_PINNdens = np.array(test_predictions_densPINN)
test_conv_PINNdens = np.reshape(np.ravel(test_conv_PINNdens),(len(dens_test_labels),9))

test_conv_smax = np.array(test_predictions_smax)
test_conv_smax = np.reshape(np.ravel(test_conv_smax),(len(test_labels),9))
testIdx = [25, 500, 1500, 2000, 2500, 3000]
# # hf.expDataPlot(test_conv_es,test_labels,testIdx,sieveCut,['Early Stop Model'])
# # hf.expDataPlot_compare(test_conv_es,test_conv_sm,test_labels,testIdx,sieveCut,[model1,model2])
# hf.expDataPlot_compare3(test_conv_all,test_conv_sm,test_conv_PINNstde,test_labels,testIdx,sieveCut,[model1,model2,model3])

plotObj.expDataPlot_comparen([test_conv_sm,test_conv_stref,test_conv_smax,test_conv_PINNdens],test_labels,testIdx,sieveCut,[model1,model2,model3,model7])
'''

print("--- %s seconds ---" % (m1Time - start_time))
print("--- %s seconds ---" % (m2Time - m1Time))
print("--- %s seconds ---" % (m3Time - m2Time))
print("--- %s seconds ---" % (m7Time - m3Time))
print("--- %s seconds ---" % (m8Time - m7Time))
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
