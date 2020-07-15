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

import kerastuner as kt

from HelperFunction import dataManipulation
from HelperFunction import plotterClass

start_time = time.time()
sieveCut = [150,250,350,600,850,1000,1400]

# Datafile without sieves
dataFile = pd.read_csv('./PBM_simData_solidDensity.csv')
dataFileWithDensity = pd.read_csv('./PBM_simData_solidDensity.csv')

# dataFile = pd.read_csv('./PBM_dcontracted.csv')
# dataFileWithDensity = pd.read_csv('./PBM_dcontracted.csv')

# Creating class objects that are 
simMod = dataManipulation.HelperFunction(dataFile,sieveCut)
stref = dataManipulation.HelperFunction(dataFile,sieveCut)
stsmaxred = dataManipulation.HelperFunction(dataFile,sieveCut)
densMod = dataManipulation.HelperFunction(dataFileWithDensity,sieveCut)
densMod2 = dataManipulation.HelperFunction(dataFileWithDensity,sieveCut)
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
# model9 = 'PINN for density'
# model10 = 'ANN for PSD'

patience_model = 10
EPOCHS = 100
labels = ['Granule_Density','Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']

################# ONLY DATA BASED MODELS #######################################
# simple model

simpleModel, history = simMod.build_train_model_2intLayers(normed_train_dataset,train_labels,patience_model,nOuput,16,'relu',EPOCHS)
m1Time = time.time()

# # # # # Data refined models
strefmodel, stref_history = stref.build_train_model_2intLayers(red_normed_train_dataset,red_train_labels,patience_model,nOuput,16,'relu',EPOCHS)
m2Time = time.time()


smaxref_model, smaxref_history = stsmaxred.build_train_model_2intLayers(redsmax_normed_train_dataset,redsmax_train_labels,patience_model,nOuput,16,'relu',EPOCHS)
m3Time = time.time()

# # evaluate simple model
loss_sm, mae_sm, mse_sm = simpleModel.evaluate(normed_test_dataset, test_labels, verbose=0)
test_predictions_sm = simpleModel.predict(normed_test_dataset).flatten()

# # use stde refined model to predict all data
loss_stref, mae_stref, mse_stref = strefmodel.evaluate(normed_test_dataset, test_labels, verbose=0)
test_predictions_stref = strefmodel.predict(normed_test_dataset).flatten()

# # use stde smax model to predict all data
loss_smax, mae_smax, mse_smax = smaxref_model.evaluate(normed_test_dataset, test_labels, verbose=0)
test_predictions_smax = smaxref_model.predict(normed_test_dataset).flatten()

# Calculate R^2
r2model_sm = simMod.calculateR2(test_predictions_sm, test_labels,labels)
r2model_stref = stref.calculateR2(test_predictions_stref, test_labels,labels)
r2model_smax = stsmaxred.calculateR2(test_predictions_smax, test_labels,labels)

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
EPOCHS = 200

###### NEW PINN #############
'''
PINNdens_model, PINNdens_history = densMod.build_train_PINN_l2_dropout(dens_normed_train_dataset,dens_train_labels,patience_model,nOutDensity,4,'tanh',EPOCHS)
m7Time = time.time()
'''

PINNdens_model_32, PINNdens_history = densMod.build_train_PINN_l1_dropout(dens_normed_train_dataset,dens_train_labels,patience_model,nOutDensity,8,'relu',EPOCHS)
m8Time = time.time()

 
########### 
dens_test_labels1 = test_labels.copy()

# use density PINN to predict data
label1 = dens_test_labels1['Granule_density']
labels = ['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']

label2 = pd.DataFrame([dens_test_labels1.pop(i) for i in labels]).T
###########

# loss_densPINN, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8, loss9, mse1, mse2, mse3, mse4, mse5, mse6, mse7, mse8, mse9 = PINNdens_model.evaluate(normed_test_dataset, [label1, label2['Bin1'],label2['Bin2'],label2['Bin3'],label2['Bin4'],label2['Bin5'],label2['Bin6'],label2['Bin7'],label2['Coarse']], verbose=1)

# loss_densPINN, loss1, loss2, mae1, mae2, mse1, mse2 = PINNdens_model.evaluate(normed_test_dataset, [label1, label2], verbose=1)

# test_predictions_densPINN = PINNdens_model_32.predict(normed_test_dataset)
# test_predictions_densPINN = np.concatenate((test_predictions_densPINN[0].flatten(),test_predictions_densPINN[1].flatten()))

# test_predictions_densPINN = np.concatenate((test_predictions_densPINN[0].flatten(),test_predictions_densPINN[1].flatten(),test_predictions_densPINN[2].flatten(),test_predictions_densPINN[3].flatten(),test_predictions_densPINN[4].flatten(),test_predictions_densPINN[5].flatten(),test_predictions_densPINN[6].flatten(),test_predictions_densPINN[7].flatten(),test_predictions_densPINN[8].flatten()),axis=0) 

# r2model_PINN = densMod.calculateR2(test_predictions_densPINN, test_labels,test_labels.keys())



# use density PINN trained data model
loss_dens, loss3_dens, loss4_dens, mae_dens, mse_dens, mae4_dens, mse4_dens = PINNdens_model_32.evaluate(dens_normed_test_dataset, [label1,label2], verbose=0)
test_predictions_dens = PINNdens_model_32.predict(dens_normed_test_dataset)
# test_predictions_dens = np.concatenate((np.reshape(test_predictions_dens[0]),(len(label1),1),np.reshape(np.ravel(test_predictions_dens[1]),len(label1)*len(label2.keys()))))

a = np.array(test_predictions_dens[0])
b = np.array(test_predictions_dens[1])
d = np.zeros((len(a),len(b[0])+1))

for i in range(len(a)):
    d[i,0] = a[i]
    for j in range(1,len(b[0])+1):
        d[i,j] = b[i,j-1]

test_predictions_dens = d.flatten()



r2model_PINN = densMod.calculateR2(test_predictions_dens, test_labels,test_labels.keys())


'''
# # use all data model to predict small data


# r2model_PINNstde = simMod.calculateR2(test_predictions_PINNstde, test_labels)
# r2model_PINNboth = simMod.calculateR2(test_predictions_PINNboth, test_labels)
r2model_dens = simMod.calculateR2(test_predictions_dens, dens_test_labels)
'
############# PREDICTING DENSITY AND PSD SEPARATELY #####################
dens_train_labels2 = dens_train_labels.copy()
dens_train_labels1 = dens_train_labels.copy()

# use density PINN to predict data
train_label1 = dens_train_labels2['Granule_density']
labels = ['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']

train_label2 = pd.DataFrame([dens_train_labels2.pop(i) for i in labels]).T

EPOCHS = 50

densModel, densHistory = densMod.build_train_densPINN_l2_dropout(dens_normed_train_dataset, train_label1,patience_model,8,'relu',EPOCHS)

densPredict = densModel.predict(dens_normed_train_dataset)
dens_normed_train_dataset1 = dens_normed_train_dataset.copy()
dens_normed_train_dataset1['Granule_density'] = np.array(densPredict)
EPOCHS = 400 
# psdModel, psdHistory = densMod.build_train_psdPI NN_l2_dropout(dens_normed_train_dataset, dens_train_labels1,patience_model,16,'relu',EPOCHS)

psdModel, psdHistory = densMod2.build_train_psdPINN_2intLayers(dens_normed_train_dataset1, train_label2,patience_model,8,32,'relu',EPOCHS)

  
###### Predicting test dataset ####

dens_test_labels1 = test_labels.copy()
# use density PINN to predict data
labels2 = ['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
test_label1 = dens_test_labels1['Granule_density']
test_label2 = pd.DataFrame([dens_test_labels1.pop(i) for i in labels]).T


loss_densMod, mae_densMod, mse_densMod = densModel.evaluate(normed_test_dataset, test_label1, verbose=0)
densPredictfin = densModel.predict(normed_test_dataset)
dens_normed_test_dataset1 = normed_test_dataset.copy()
dens_normed_test_dataset1['Granule_density'] = densPredictfin
loss_finMod, mae_finMod, mse_finMod = psdModel.evaluate(dens_normed_test_dataset1, test_label2, verbose=0)
psdPredict = psdModel.predict(dens_normed_test_dataset1)


finalPrediction = np.concatenate((np.array(densPredictfin.flatten()),np.array(psdPredict.flatten())))

r2model_densPINN = densMod.calculateR2(densPredictfin.flatten(), test_label1,['Granule_Density'])
r2model_ANNPSD = densMod2.calculateR2(psdPredict.flatten(),test_label2,labels2)
'''
print("\nTest set Loss(MSE), MAE and R^2 of ", model1 ,"is", loss_sm, mae_sm, r2model_sm)
print("Test set Loss(MSE), MAE and R^2 of ", model2 ,"is",  loss_stref, mae_stref, r2model_stref)
print("Test set Loss(MSE), MAE and R^2 of ", model3 ,"is",  loss_smax, mae_smax, r2model_smax)
# print("Test set Loss(MSE), MAE and R^2 of ", model4 ,"is",  loss_PINNstde, mae_PINNstde, r2model_PINNstde)
# print("Test set Loss(MSE), MAE and R^2 of ", model5 ,"is",  loss_PINNboth, mae_PINNboth, r2model_PINNboth)
# print("Test set Loss(MSE) and R^2 of ", model7 ,"is",  loss_densPINN, r2model_PINN)
print("Test set Loss(MSE) and R^2 of ", model8 ,"is",  loss_dens, r2model_PINN)
# print("Test set Loss(MSE), MAE and R^2 of ", model8 ,"is",  loss_dens, mae_dens, r2model_dens)
# print("Test set Loss(MSE), MAE and R^2 of ", model9 ,"is",  (loss_densMod), (mae_densMod), r2model_densPINN)
# print("Test set Loss(MSE), MAE and R^2 of ", model10 ,"is",  (loss_finMod), (mae_finMod), r2model_ANNPSD)


# print("All data set Loss, MAE and MSE of ", model2 ,"is",  loss_all, mae_all, mse_all)
# print("Small data set Loss, MAE and MSE of ", model2 ,"is",  loss_small, mae_small, mse_small)

#### saving model


# simpleModel.save('AllData_2000')
# strefmodel.save('StDErefinedModel_2000')
# smaxref_model.save('StDE_SMax_refinedModel_2000')
# # PINNdens_model.save('PINN_16_2000')
# # PINNdens_model_32.save('PINN_dropout_2000')
# psdModel.save('psdModel')
# densModel.save('densModel') 

print("--------All models were saved successfully--------")

# All plots start here
# comparing 3 models
plt.figure()
plotObj.history_plotter_comparen_noval([history, stref_history,smaxref_history,PINNdens_history], 'loss', [model1, model2, model3,model8])

# plt.figure()
# plotObj.parityPlot_dens(test_labels, test_predictions_densPINN, model7)

# plt.figure()
# plotObj.parityPlot_dens(test_labels, finalPrediction, model9)

# plt.figure()
# plotObj.parityPlot(test_labels, finalPrediction, model9)

# plt.figure()
# plotObj.parityPlot(test_labels, test_predictions_densPINN, model7)

## normed_test_dataset, test_labels

# plt.figure()
# plotObj.parityPlot(test_labels, test_predictions_sm,model1)
# plt.figure()
# plotObj.parityPlot(test_labels, test_predictions_stref, model2)
# plt.figure()
# plotObj.parityPlot(test_labels, test_predictions_dens, model8)
plt.figure()
plotObj.parityPlot(test_labels, test_predictions_dens, model8)
# plt.figure()
# plotObj.parityPlot(test_labels, test_predictions_smax, model3)

# plt.figure()
# plotObj.parityPlot_dens(test_labels, test_predictions_densPINN, model7)


# plt.figure()
# plotObj.parityPlot_dens(test_labels, test_predictions_sm,model1)
# plt.figure()
# plotObj.parityPlot_dens(test_labels, test_predictions_stref, model2)
# plt.figure()
# plotObj.parityPlot(test_labels, test_predictions_dens, model8)


# plt.figure()
# plotObj.parityPlot_dens(test_labels, test_predictions_smax, model3)

test_conv_sm = np.array(test_predictions_sm)
test_conv_sm = np.reshape(np.ravel(test_conv_sm),(len(test_labels),9))

test_conv_stref = np.array(test_predictions_stref)
test_conv_stref = np.reshape(np.ravel(test_conv_stref),(len(test_labels),9))

# test_conv_dens = np.array(test_predictions_dens)
# test_conv_dens = np.reshape(np.ravel(test_conv_dens),(len(dens_test_labels),9))

test_conv_PINNdens = np.array(test_predictions_dens)
test_conv_PINNdens = np.reshape(np.ravel(test_conv_PINNdens),(len(dens_test_labels),9))

test_conv_smax = np.array(test_predictions_smax)
test_conv_smax = np.reshape(np.ravel(test_conv_smax),(len(test_labels),9))


testIdx = [25, 500, 1500, 2000, 2500, 3000]
# # hf.expDataPlot(test_conv_es,test_labels,testIdx,sieveCut,['Early Stop Model'])
# # hf.expDataPlot_compare(test_conv_es,test_conv_sm,test_labels,testIdx,sieveCut,[model1,model2])
# hf.expDataPlot_compare3(test_conv_all,test_conv_sm,test_conv_PINNstde,test_labels,testIdx,sieveCut,[model1,model2,model3])

# plotObj.expDataPlot_comparen([test_conv_sm,test_conv_stref,test_conv_smax,test_conv_PINNdens],test_labels,testIdx,sieveCut,[model1,model2,model3,model7])


# test_conv_dens = np.array(finalPrediction)
# test_conv_dens = np.reshape(np.ravel(test_conv_dens),(len(dens_test_labels),9))

# test_conv_PINNdens = np.array(finalPrediction)
# test_conv_PINNdens = np.reshape(np.ravel(test_conv_PINNdens),(len(dens_test_labels),9))

testIdx = [25, 200, 500, 1000, 1200, 3000]
plotObj.expDataPlot_comparen([test_conv_sm,test_conv_stref,test_conv_smax,test_conv_PINNdens],test_labels,testIdx,sieveCut,[model1,model2,model3,model7])
'''
# print("--- %s seconds ---" % (m1Time - start_time))
# print("--- %s seconds ---" % (m2Time - m1Time))
# print("--- %s seconds ---" % (m3Time - m2Time))
# print("--- %s seconds ---" % (m7Time - m3Time))
# print("--- %s seconds ---" % (m8Time - m7Time))
# print("--- %s seconds ---" % (time.time() - start_time))
'''
plt.show()