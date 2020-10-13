# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:08:45 2020

@author: Chaitanya Sampat

This module contains all data manipulation for PINN model

"""
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
from tensorflow import keras
import talos
# import tensorflow_docs as tfdocs

from tensorflow_docs import modeling
from math import exp
from keras import regularizers, layers, initializers
from keras.layers import Dense, Input, Concatenate, Dropout, Conv2D
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, RMSprop, Nadam
from keras.initializers import glorot_normal, normal

from talos.model.normalizers import lr_normalizer

class HelperFunction:
    def __init__(self, dataFile, sieveCut, frac=0.8, yieldStrength=1e4):
        self.sieveCut = sieveCut
        self.dataFile = dataFile
        self.yieldStrength = yieldStrength
        self.frac = frac
        self.normed_train_dataset, self.train_labels, self.normed_test_dataset, self.test_labels = self.normedDataSplitWithDensity(self.dataFile,self.frac)
        self.trainingdata = pd.concat([self.normed_train_dataset, self.train_labels], axis=1)


    def norm(self, x, train_stats):
        return (x - train_stats['min']) / (train_stats['max'] - train_stats['min'])
        #return (x - train_stats['mean']) / train_stats['std']
    
    def norm_dens(self,x):
        return (x / 1000)

    
    def normedDataSplit(self, dataFile, frac=0.8, random_state=42):
        train_dataset = dataFile.sample(frac,random_state)
        test_dataset = dataFile.drop(train_dataset.index)
        
        labels = ['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
        
        train_labels = pd.DataFrame([train_dataset.pop(i) for i in labels]).T
        test_labels = pd.DataFrame([test_dataset.pop(i) for i in labels]).T
        
        train_stats = train_dataset.describe()
        train_stats = train_stats.transpose()
        
        test_stats = test_dataset.describe()
        test_stats = test_stats.transpose()
        
        normed_train_dataset = self.norm(train_dataset, train_stats)
        normed_test_dataset = self.norm(test_dataset, test_stats)
        
        return normed_train_dataset, train_labels, normed_test_dataset, test_labels
    
    def normedDataSplitWithDensity(self, dataFile, frac=0.8, random_state=0):
        train_dataset = dataFile.sample(frac=0.8,random_state=42)
        test_dataset = dataFile.drop(train_dataset.index)
        
        labels = ['Granule_density','Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
        
        train_labels = pd.DataFrame([train_dataset.pop(i) for i in labels]).T
        test_labels = pd.DataFrame([test_dataset.pop(i) for i in labels]).T
        
        train_stats = train_dataset.describe()
        train_stats = train_stats.transpose()
        
        test_stats = test_dataset.describe()
        test_stats = test_stats.transpose()
        
        normed_train_dataset = self.norm(train_dataset, train_stats)
        normed_test_dataset = self.norm(test_dataset, test_stats)

        train_label_stats = train_labels['Granule_density'].describe()
        train_label_stats = train_label_stats.transpose()
        

        test_label_stats = test_labels['Granule_density'].describe()
        test_label_stats = test_label_stats.transpose()

        train_labels['Granule_density'] = self.norm_dens(train_labels['Granule_density'])
        test_labels['Granule_density'] = self.norm_dens(test_labels['Granule_density'])

        return normed_train_dataset, train_labels, normed_test_dataset, test_labels

    def normedDataLabelSplit(self, dataFile):
        
        dataSet = dataFile.copy()
        labels = ['Granule_density','Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
        
        labelData = pd.DataFrame([dataSet.pop(i) for i in labels]).T
        # test_labels = pd.DataFrame([test_dataset.pop(i) for i in labels]).T
        
        data_stats = dataSet.describe()
        data_stats = data_stats.transpose()
        
        # test_stats = test_dataset.describe()
        # test_stats = test_stats.transpose()
        
        dataSet = self.norm(dataSet, data_stats)
        # normed_test_dataset = self.norm(test_dataset, test_stats)

        labelDataStats = labelData['Granule_density'].describe()
        labelDataStats = labelDataStats.transpose()
        

        # test_label_stats = test_labels['Granule_density'].describe()
        # test_label_stats = test_label_stats.transpose()

        labelData['Granule_density'] = self.norm_dens(labelData['Granule_density'])
        # test_labels['Granule_density'] = self.norm_dens(test_labels['Granule_density'])

        return dataSet, labelData

############# ANN MODELS ##################################


    def get_callbacks(self,pat=10):
      return [modeling.EpochDots(),
              tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=pat)]
    
    def build_train_model_2intLayers(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=16, actFn='relu', EPOCHS=100,learnRate=0.001,momentumN=0.0,lastActFn='linear',regCons=0.0,doRate=0.0):
        model = Sequential([
            Dense(nNodes, activation=actFn, input_shape=[len(normed_train_dataset.keys())]),
            Dense(nNodes, activation=actFn),
            Dense(nOutput, activation=lastActFn)])
    
        model.compile(optimizer=SGD(learning_rate=learnRate,momentum=momentumN, nesterov=False),
                  loss='mse', metrics = ['mae','mse'])
        print(model.summary())
        history = model.fit(normed_train_dataset, train_labels, epochs=EPOCHS, 
                            validation_split = 0.33, verbose=0, callbacks=self.get_callbacks(patience_model))
        
        return model, history

    def build_train_model_3intLayers(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=8, actFn='relu', EPOCHS=100):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(nNodes, activation=actFn, input_shape=[len(normed_train_dataset.keys())]),
            tf.keras.layers.Dense(nNodes, activation=actFn),
            tf.keras.layers.Dense(nNodes, activation=actFn),
            tf.keras.layers.Dense(nOutput, activation='relu')])
    
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.0, nesterov=False),
                  loss='mse', metrics = ['mae','mse'])
        print(model.summary())
        history = model.fit(normed_train_dataset, train_labels, epochs=EPOCHS, 
                            validation_split = 0.33, verbose=0, callbacks=self.get_callbacks(patience_model))
        
        return model, history
    
    def build_train_regularizedModel_2intLayers(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=8, actFn='relu', EPOCHS=100):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(nNodes, activation=actFn, 
                                  input_shape=[len(normed_train_dataset.keys())], 
                                  activity_regularizer=regularizers.l2(0.001),),
            tf.keras.layers.Dense(nNodes, activation=actFn, activity_regularizer=regularizers.l2(0.001),),
            tf.keras.layers.Dense(nOutput, activation='relu')])
    
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.0, nesterov=False),
                  loss='mse', metrics = ['mae','mse'])
        print(model.summary())
        history = model.fit(normed_train_dataset, train_labels, epochs=EPOCHS, 
                            validation_split = 0.33, verbose=0, callbacks=self.get_callbacks(patience_model))
        
        return model, history
    
    
    def build_train_PINNstde(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=8, actFn='relu', EPOCHS=100):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(nNodes, activation=actFn, 
                                  input_shape=[len(normed_train_dataset.keys())], 
                                  activity_regularizer=regularizers.l2(0.001),),
            tf.keras.layers.Dense(nNodes, activation=actFn, activity_regularizer=regularizers.l2(0.001),),
            tf.keras.layers.Dense(nOutput, activation='relu')])
    
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.0, nesterov=False),
                  loss=self.lossFunc_StdeOnly, metrics = ['mae','mse'])
        print(model.summary())
        history = model.fit(normed_train_dataset, train_labels, epochs=EPOCHS, 
                            validation_split = 0.33, verbose=0, callbacks=self.get_callbacks(patience_model))
        
        return model, history
    
    def build_train_PINNStdeSmax(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=8, actFn='relu', EPOCHS=100):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(nNodes, activation=actFn, 
                                  input_shape=[len(normed_train_dataset.keys())]),
            tf.keras.layers.Dense(nNodes, activation=actFn),
            tf.keras.layers.Dense(nNodes, activation=actFn),
            tf.keras.layers.Dense(nOutput, activation='relu')])
    
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.0, nesterov=False),
                  loss=self.lossFunc_StdeSmax, metrics = ['mae','mse'])
        print(model.summary())
        history = model.fit(normed_train_dataset, train_labels, epochs=EPOCHS, 
                            validation_split = 0.33, verbose=0, callbacks=self.get_callbacks(patience_model))
        
        return model, history
    
    def build_train_PINNstde3(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=8, actFn='relu', EPOCHS=100):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(nNodes, activation=actFn, 
                                  input_shape=[len(normed_train_dataset.keys())]),
            tf.keras.layers.Dense(nNodes, activation=actFn),
            tf.keras.layers.Dense(nNodes, activation=actFn),
            tf.keras.layers.Dense(nOutput, activation='relu')])
    
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.0, nesterov=False),
                  loss=self.lossFunc_StdeOnly, metrics = ['mae','mse'])
        print(model.summary())
        history = model.fit(normed_train_dataset, train_labels, epochs=EPOCHS, 
                            validation_split = 0.0, verbose=0, callbacks=self.get_callbacks(patience_model))
        
        return model, history
    
    def build_train_PINNStdeSmax3(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=8, actFn='relu', EPOCHS=100):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(nNodes, activation=actFn, 
                                  input_shape=[len(normed_train_dataset.keys())], 
                                  activity_regularizer=regularizers.l2(0.001),),
            tf.keras.layers.Dense(nNodes, activation=actFn, activity_regularizer=regularizers.l2(0.001),),
            tf.keras.layers.Dense(nOutput, activation='relu')])
    
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.0, nesterov=False),
                  loss=self.lossFunc_StdeSmax, metrics = ['mae','mse'])
        print(model.summary())
        history = model.fit(normed_train_dataset, train_labels, epochs=EPOCHS, 
                            validation_split = 0.33, verbose=0, callbacks=self.get_callbacks(patience_model))
        
        return model, history
       
    def build_train_PINN_mul(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=16, actFn='relu', EPOCHS=100):

        regCons = 0.001
        input_layer = Input(shape=(6,))
        dense_1 = Dense(nNodes, activation='sigmoid', activity_regularizer=regularizers.l1(regCons))(input_layer)
        dense_2 = Dense(nNodes, activation='sigmoid', activity_regularizer=regularizers.l1(regCons))(dense_1)
        dense_3 = Dense(nNodes, activation='sigmoid', activity_regularizer=regularizers.l1(regCons))(dense_2)
        dense_4 = Dense(nNodes, activation='sigmoid', activity_regularizer=regularizers.l1(regCons))(dense_3)
        dense_5 = Dense(nNodes, activation='sigmoid', activity_regularizer=regularizers.l1(regCons))(dense_4)
        dense_6 = Dense(nNodes, activation='sigmoid', activity_regularizer=regularizers.l1(regCons))(dense_5)
        # separating outputs for density and GSD for custom loss function
        output_1 = Dense(1,activation='sigmoid',name='out1', activity_regularizer=regularizers.l1(regCons))(dense_6)
        # output_2 = Dense(8,activation='sigmoid',name='out2')(dense_2)
        output_2 = Dense(8,activation=self.mapping_to_target_range,name='out2', activity_regularizer=regularizers.l1(regCons))(dense_6)      
        # output_2 = Dense(8,activation=self.mapping_to_target_range,name='out2')(dense_4) 
        
        train_labels_copy = pd.DataFrame.copy(train_labels)
        label1 = train_labels['Granule_density']
        labels = ['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
        
        # label2 = ['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
        label2 = pd.DataFrame([train_labels_copy.pop(i) for i in labels]).T
        model = Model(inputs=[input_layer], outputs=[output_1,output_2])

        model.compile(optimizer=SGD(learning_rate=0.01,momentum=0.001, nesterov=False),loss=self.lossFunc_DensityStde(output_1), metrics = ['mae','mse'])
        # model.compile(optimizer=Adam(learning_rate=0.001,beta_1=0.1,beta_2=0.2,epsilon=0.01,amsgrad=True),loss=self.lossFunc_DensityStde(output_1), metrics = ['mae','mse'])
        
        w1 = np.full(len(self.normed_train_dataset),10)
        w2 = np.full(len(self.normed_train_dataset),1)
        print(model.summary())
        history = model.fit(normed_train_dataset, [label1, label2], epochs=EPOCHS, 
                            verbose=1, validation_split=0.2,sample_weight={'out1': w1, 'out2':w2},use_multiprocessing=True)
                
        return model, history


    def build_train_PINN_l2_dropout(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=16, actFn='relu', EPOCHS=100,learnRate=0.001,momentumN=0.0,lastActFn='linear',regCons=0.0,doRate=0.0):
        input_layer = Input(shape=(len(normed_train_dataset.keys()),))
        dense_1 = Dense(nNodes, activation=actFn,activity_regularizer=regularizers.l2(regCons))(input_layer)
        # dense_1 = Dense(nNodes, activation=actFn)(input_layer)
        dense_1 = Dropout(doRate)(dense_1)
        dense_2 = Dense(nNodes, activation=actFn,activity_regularizer=regularizers.l2(regCons))(dense_1)
        # dense_2 = Dense(nNodes, activation=actFn)(dense_1)
        dense_2 = Dropout(doRate)(dense_2)
        dense_3 = Dense(nNodes, activation=actFn, activity_regularizer=regularizers.l2(regCons))(dense_2)
        # dense_3 = Dense(nNodes, activation=actFn)(dense_2)
        dense_3 = Dropout(doRate)(dense_3)
        # dense_4 = Dense(nNodes, activation=actFn,activity_regularizer=regularizers.l2(regCons))(dense_3)
        # dense_4 = Dense(nNodes, activation=actFn)(dense_3)
        # dense_4 = Dropout(doRate)(dense_4)
        # dense_5 = Dense(nNodes, activation=actFn,activity_regularizer=regularizers.l1(regCons))(dense_4)
        # # dense_5 = Dropout(doRate)(dense_5)
        # dense_6 = Dense(nNodes, activation=actFn, activity_regularizer=regularizers.l1(regCons))(dense_5)
        # dense_6 = Dropout(doRate)(dense_6)
        # dense_11 = Dropout(doRate)(dense_11)
        # separating outputs for density and GSD for custom loss function
        # output_1 = Dense(1,activation=self.mapping_to_target_range_density,name='Density', activity_regularizer=regularizers.l2(regCons))(dense_3)
        output_1 = Dense(1,activation=lastActFn,name='Density', activity_regularizer=regularizers.l2(regCons))(dense_3)
        output_2 = Dense(8,activation=lastActFn,name='GSD')(dense_3) 
        # output_2 = Dense(8,activation=self.mapping_to_target_range,name='GSD')(dense_3) 
        
        train_labels_copy = pd.DataFrame.copy(train_labels)
        label1 = train_labels['Granule_density']
        labels = ['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
        
        # label2 = ['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
        label2 = pd.DataFrame([train_labels_copy.pop(i) for i in labels]).T
        model = Model(inputs=[input_layer], outputs=[output_1,output_2])
        
        model.compile(optimizer=SGD(learning_rate=learnRate,momentum=momentumN, nesterov=True),loss=self.lossFunc_DensityStde(output_1), metrics = ['mae','mse'])
        # model.compile(optimizer=Adam(learning_rate=learnRate,amsgrad=True),loss=self.lossFunc_DensityStde(output_1), metrics = ['mae','mse'])
        
        w1 = np.full(len(self.normed_train_dataset),1)
        w2 = np.full(len(self.normed_train_dataset),1)
        print(model.summary())
        history = model.fit(normed_train_dataset, [label1, label2], epochs=EPOCHS, 
                            verbose=1, validation_split=0.33,sample_weight={'Density': w1, 'GSD':w2},use_multiprocessing=True,callbacks=self.get_callbacks(patience_model))
        

        # for layer in model.layers: print(layer.get_config(), layer.get_weights())
        return model, history



     # redefining PINN model for hyperparameter scan   
    def hyperparameter_Scan(self,normed_train_dataset, train_labels, test_data, test_labels, param):
    #patience_model, nOutput, nNodes=16, actFn='relu', EPOCHS=100,learnRate=0.001,momentumN=0.0,lastActFn='linear',regCons=0.0,doRate=0.0):
        input_layer = Input(shape=(len(normed_train_dataset.keys()),))
        dense_1 = Dense(param['nNodes'], activation=param["activation"],activity_regularizer=regularizers.l2(param["regCons"]))(input_layer)
        # dense_1 = Dropout(param["dropRate"])(dense_1)
        dense_2 = Dense(param['nNodes'], activation=param["activation"],activity_regularizer=regularizers.l2(param["regCons"]))(dense_1)
        dense_2 = Dropout(param["dropRate"])(dense_2)
        dense_3 = Dense(param['nNodes'], activation=param["activation"], activity_regularizer=regularizers.l2(param["regCons"]))(dense_2)
        # dense_3 = Dropout(param["dropRate"])(dense_3)
        dense_4 = Dense(param['nNodes'], activation=param["activation"],activity_regularizer=regularizers.l2(param["regCons"]))(dense_3)
        dense_4 = Dropout(param["dropRate"])(dense_4)
        # separating outputs for density and GSD for custom loss function
        output_1 = Dense(1,activation=param["last_activation"],name='Density', activity_regularizer=regularizers.l2(param["regCons"]))(dense_4)
        # output_2 = Dense(8,activation='sigmoid',name='out2',activity_regularizer=regularizers.l1(regCons))(dense_4)
        output_2 = Dense(8,activation=param["last_activation"],name='GSD', activity_regularizer=regularizers.l2(param["regCons"]))(dense_4)
        
        train_labels_copy = pd.DataFrame.copy(train_labels)
        label1 = train_labels['Granule_density']
        labels = ['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
        
        # label2 = ['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
        label2 = pd.DataFrame([train_labels_copy.pop(i) for i in labels]).T
        model = Model(inputs=[input_layer], outputs=[output_1,output_2])
        
        model.compile(optimizer=param['optimizer'](lr=lr_normalizer(param['learningRate'],param['optimizer'])),loss=self.lossFunc_DensityStde(output_1), metrics = ['mae','mse'])
        # model.compile(optimizer=Adam(learning_rate=0.0001,beta_1=0.1,beta_2=0.2,epsilon=0.001,amsgrad=True),loss=self.lossFunc_DensityStde(output_1), metrics = ['mae','mse'])
        
        w1 = np.full(len(self.normed_train_dataset),1)
        w2 = np.full(len(self.normed_train_dataset),1)
        # print(model.summary())
        test_label_copy = pd.DataFrame.copy(test_labels)
        tlabel1 = test_labels['Granule_density']
        tlabel2 = pd.DataFrame([test_label_copy.pop(i) for i in labels]).T

        out = model.fit(normed_train_dataset, [label1, label2], epochs=param['epochs'], 
                            verbose=0, validation_data=[test_data,[tlabel1,tlabel2]],sample_weight={'Density': w1, 'GSD':w2},use_multiprocessing=True)
        

        # for layer in model.layers: print(layer.get_config(), layer.get_weights())
        return out, model

########### PHYSICS CALCULATIONS #############################################           
    def d50FromPSDCalculator(self,PSDvals):
        sieves = len(self.sieveCut)
        sieveCut = self.sieveCut
        dataPts = len(PSDvals)
        d50 = np.zeros((dataPts,))
        
        for i in range(dataPts):
            lowID = 0
            highID = 1
            # to calculate d50 we need the sieve value at 0.5
            for j in range(sieves):
                if(PSDvals[i,j] < 0.5):
                    lowID = j
                    if(j<6):
                        highID = j+1
                    else:
                        highID=j
                        lowID=j-1
            
            # matching slopes to get the intermidiate value at 0.5
            if (highID == lowID):
                d50[i] = sieveCut[highID]    
            else:
                d = (PSDvals[i,highID] - PSDvals[i,lowID]) # denominator
                if (d==0):
                    d50[i] = sieveCut[lowID]    
                else:
                    m = sieveCut[highID] * ((0.5 - PSDvals[i,lowID]) / d) #  term1
                    n = sieveCut[lowID] * ((PSDvals[i,highID] - 0.5) / d) # second term
                    
                    d50[i] = m + n
            
        d50[d50<0] = sieveCut[0]
        return d50
        
    
    def stdeCalculation(self,dataFile):
        particleDensity = (np.array(dataFile['Granule_density']))
        dImp = np.array(dataFile['impellerDiameter'])
        rpm = np.array(dataFile['rpm'])
        bins = np.array((dataFile[['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']]).values)
        yS = np.full((len(particleDensity),), 2 * self.yieldStrength)
        l = (len(particleDensity),)
        a = np.full(l,((np.pi * 0.15) / 60))
        # nParticles = np.divide(particleDensity,((4/3)*np.pi*np.power(d50/2,3)))
        Uc = np.multiply(a,np.multiply(rpm,dImp))# shear=pi*rpm*diameter/60
        # Uc = np.multiply(d50/2,shear)
        StDe = np.divide(np.multiply(particleDensity,(np.square(Uc))), yS)
        dataFile['StDe'] = StDe
        # self.trainingdata['StDe'] = StDe
        return dataFile
    
    def smaxCalculation(self,dataFile):
        particleDensity = np.array(dataFile['solidDensity']) 
        solidWt = np.array(dataFile['batch_amount']) / 1000
        liqWt = np.array(dataFile['liq_amount']) / 100
        # iniPor = np.array(self.dataFile['Initial_Porosity'])
        minPorVal = 0.18
        minPor = np.ones(len(liqWt,)) * minPorVal
        liqDen = 1000
        n1 = (1 - minPor) # part of numerator
        d1 = liqDen * minPor # demonimator
        w = np.divide(solidWt,liqWt)
        num = np.multiply(w,np.multiply(particleDensity,n1))
        smax = np.divide(num,d1)
        dataFile['Smax'] = smax
        # self.trainingdata['Smax'] = smax
        return dataFile
        
        
            
    def stdeBasedDataRemoval(self,dataFile):
        dataFile1 = self.stdeCalculation(dataFile)
        # dataFile1 = self.trainingdata
        dataFile1 = dataFile1.dropna()
        dataFile1 = dataFile1[dataFile1['StDe'] <= (0.1)]  
        dataFile1 = dataFile1.drop(columns=['StDe'])
        # dataFile1 = dataFile1.drop(columns=['d50'])
        # self.trainingdata.drop(columns=['StDe'])
        # self.trainingdata.drop(columns=['d50'])
        return dataFile1
    
    
    def stdesmaxDataRemoval(self,dataFile):
      
        dataFile1 = self.stdeCalculation(dataFile)
        dataFile1 = self.smaxCalculation(dataFile)
        # dataFile1 = self.trainingdata
        dataFile1 = dataFile1.dropna()
        dataFile1 = dataFile1[dataFile1['StDe'] <= (0.1)]
        dataFile1 = dataFile1[dataFile1['Smax'] <= 1]
        dataFile1 = dataFile1.drop(columns=['StDe'])
        dataFile1 = dataFile1.drop(columns=['Smax'])
        # dataFile1 = dataFile1.drop(columns=['d50'])
        return dataFile1
    
    
   
    def lossFunc_StdeSmax(self,yTrue,yPred):
        dataFile1 = self.stdeCalculation() # get values StDE calculated
        stdeVal = np.array(dataFile1['StDe']) # separating out the StDe values into an numpy array
        stError = stdeVal[stdeVal>0.1]
        stError = (stError - 0.1) / len(stError)
        dataFile1 = self.smaxCalculation()
        smax = np.array(dataFile1['Smax'])
        smaxError = smax[smax<0.5]
        smaxError = (0.5 - smaxError) / len(smaxError)
        addError = K.mean(stError) + K.mean(smaxError)
        
        return K.mean(K.square(yTrue - yPred)) + addError
       
    def lossFunc_DensityStde(self,output_1):
        def loss(yTrue,yPred):
            # rpm = np.array(self.normed_train_dataset['rpm'])
            # granuleDensity = K.slice(yPred, 0, len(rpm))
            # granuleDensity = granuleDensity.numpy()
            
            Uc, Ys = self.stdePreCalc()
            
            Uc = np.array(Uc)
            Ys = np.array(Ys)
            b = np.power(Uc,2)
            c = K.cast_to_floatx(np.multiply(Ys,b))
            d = K.cast_to_floatx(np.full(len(Uc),(0.1)))
            rho_comp = K.cast_to_floatx(np.divide(np.divide(d,c),1e3))
            rho_l = rho_comp - output_1
            rho_l = rho_l[rho_l < 0]
            length = K.shape(rho_l)[0]
            length = K.cast(length,K.floatx())
            length = K.reshape(length,(1,1))
            addError = K.mean(K.square(rho_l))

            # stde_output = K.dot(output_1,K.transpose(c))
            # stde_comp = stde_output - d
            # stde_v = stde_comp[stde_comp > 0]

            # addError = k.mean(stde_v)
                    
            return K.mean(K.square(yTrue - yPred)) + addError
        return loss
              
    def stdePreCalc(self):
       dImp = (np.array(self.trainingdata['impellerDiameter'])*0.625) + 0.125
       rpm = (np.array(self.trainingdata['rpm'])*500) + 100
       m = np.multiply(rpm,dImp)
       l = len(rpm)
       a = np.full(l,((np.pi * 0.15) / 60))
       Uc = np.multiply(m,a)
       ys = np.full(l,(1 / (2 * self.yieldStrength)))
       return Uc, ys
        
    
    def calculateR2(self, y_predicted, y_true, labels):
        # labels = ['Granule_Density','Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
        y_predicted_mean = np.array(y_predicted).mean()
        y_true_reshaped = np.reshape(np.ravel(y_true),(len(y_true)*len(labels),))
        '''
        test['output_pred'] = model.predict(x=np.array(test[inputs]))
        output_mean = test['output'].mean()    # Is this the correct mean value for r2 here?
        test['SSres'] = np.square(test['output']-test['output_pred'])
        test['SStot'] = np.square(test['output']-output_mean)
        r2 = 1-(test['SSres'].sum()/(test['SStot'].sum()))
        '''    
        SSres = np.square(y_true_reshaped - y_predicted)
        SStol = np.square(y_true_reshaped - y_predicted_mean)
        
        r2 = 1 - (SSres.sum() / SStol.sum())          
        return r2
        
    def mapping_to_target_range(self, x, target_min=0.02, target_max=1.00):
        x02 = (x)
        scale = (target_max-target_min)
        return  (x02) * scale + target_min
        # return  x02 + target_min
      
    def mapping_to_target_range_density(self, x, target_min=0.3, target_max=0.7):
        x02 = (x)
        scale = (target_max-target_min)
        return  (x02) * scale + target_min
        
# dataFile = pd.read_csv('./PBM_simData_noSieve.csv')    

# bins = np.array((dataFile[['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']]).values)
# sieveCut = [150,250,350,600,850,1000,1400]
# a = HelperFunction(dataFile,sieveCut,1e5)
# st = a.stdeCalculation()
# st1 = a.smaxCalculation()

# st1.to_csv('inputdata.csv')

# a = HelperFunction(dataFile,sieveCut,1e5)
# p = [1,2,5,4,1,6,8]
# s = [1,2,4,3,2,5,8]
# r2 = a.calculateR2(p, s)
# print(r2)